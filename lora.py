"""lora.py — find LoRa transmissions in IQ samples and measure them.

DSP only, no hardware, so it can be tested on synthetic signals and run
on saved recordings. lorascan.py does the tuning and printing.

How LoRa looks
--------------
LoRa is chirp spread spectrum. Each symbol is a tone that sweeps across
the whole channel bandwidth (BW: 62.5, 125, 250 or 500 kHz) in 2**SF / BW
seconds, where SF, the spreading factor, is 7 to 12. A packet starts
with a preamble of identical up-chirps (8 for LoRaWAN, 16 for
Meshtastic), then two sync symbols and 2.25 down-chirps, then the data.
A transmitter can also invert I/Q, which turns every chirp around:
LoRaWAN gateways send downlinks that way so that nodes and gateways
don't hear each other.

How this module finds it
------------------------
1. find_bursts(): a spectrogram (512-point frames, ~0.2 ms each at 2.4
   MS/s), each bin divided by its own noise floor (a low percentile over
   time, so a long packet doesn't raise it). Cells well above the floor
   are "active". A chirp is a diagonal line on the spectrogram, so the
   active cells are bridged in time across one longest symbol (SF12 at
   125 kHz, 33 ms) and labelled into connected regions. Each region at
   least MIN_WIDTH_HZ wide and MIN_BURST_S long is a burst. Its occupied
   bandwidth is measured off the burst's average spectrum (a chirp's is
   flat-topped) and snapped to the nearest LoRa bandwidth.
2. chirp_facts(): the textbook LoRa trick. The burst is moved to 0 Hz
   and resampled to exactly BW, so one symbol is 2**SF samples; multiplied
   by the conjugate of an up-chirp of the right SF, every preamble chirp
   becomes a single tone, i.e. one sharp FFT peak. Trying each SF, both
   ways round, the sharpest peak gives the SF and the chirp direction.
   Noise, FSK and other signals give no sharp peak.

Burst slope alone can't tell SF7/125k from SF9/250k (same Hz per
second); the measured bandwidth settles that, which is why BW is
measured first.
"""

from fractions import Fraction

import numpy as np
from scipy import fft as sp_fft
from scipy import ndimage
from scipy.signal import resample_poly

LORA_BANDWIDTHS = (62_500, 125_000, 250_000, 500_000)
SPREADING_FACTORS = tuple(range(7, 13))

SAMPLE_RATE = 2_400_000         # the fastest an RTL-SDR runs without drops.
FRAME = 512                     # spectrogram frame: 4.7 kHz, 0.21 ms.
COARSE = 4                      # detect on 4 x 4 averaged cells (19 kHz,
                                # 0.85 ms); measure on the full grid.
FLOOR_PERCENTILE = 25           # noise floor per bin, robust to busy channels.
BOX_HZ = 62_500                 # detect on power averaged over 62.5 kHz
BOX_S = 0.004                   # by 4 ms: catches any LoRa sweep speed.
DEFAULT_THRESHOLD_DB = 1.5      # box average must reach this over noise ...
GROW_DB = 0.6                   # ... and the burst is everything connected
                                # to that above this (hysteresis).
MERGE_GAP_S = 0.030             # pieces of one weak packet closer than this
                                # on the same channel are joined.
BRIDGE_S = 0.005                # dips shorter than this don't end a burst.
MIN_BURST_S = 0.002
MIN_WIDTH_HZ = 40_000           # well under 62.5 kHz isn't LoRa.
SPEC_SMOOTH_BINS = 5            # ~23 kHz smoothing when measuring width.
ON_SMOOTH_FRAMES = 9            # ~2 ms smoothing when timing a burst.
GAP_HZ = 25_000                 # spectrum dips narrower than this are bridged.
WIDTH_DROP_DB = 6.0             # bandwidth measured at -6 dB.
SNAP_TOLERANCE = 0.25           # measured width within 25% of a LoRa BW.
WINDOWS = 8                     # preamble symbols summed per SF test.
MIN_WINDOWS = 4                 # fewer than this and an SF isn't tried.
CHIRP_THRESHOLD_DB = 11.0       # dechirped peak over the average bin pair.
                                # Noise scores ~4 dB and real FSK bursts
                                # off the air up to ~9; LoRa scores 14+.
MIN_PACKET_SYMBOLS = 10         # preamble (6+) + sync (4.25) alone is more.

_BIN_HZ = SAMPLE_RATE / FRAME
GAP_BINS = int(GAP_HZ / _BIN_HZ)
MIN_WIDTH_BINS = int(np.ceil(MIN_WIDTH_HZ / _BIN_HZ))


# --------------------------------------------------------------------------
# Chirps
# --------------------------------------------------------------------------

def upchirp(sf, samples_per_chip=1, symbol=0):
    """One LoRa up-chirp at baseband, BW * samples_per_chip samples/s.

    Sweeps -BW/2 .. +BW/2; `symbol` starts the sweep that many chips up
    and wraps, which is how LoRa carries data.
    """
    n_chips = 2 ** sf
    n = np.arange(n_chips * samples_per_chip) / samples_per_chip
    # Instantaneous frequency, in cycles per chip, is
    # ((n + symbol) mod N) / N - 1/2: integrate it, dropping one cycle per
    # chip after the wrap.
    wrapped = np.maximum(0.0, n - (n_chips - symbol))
    phase = 2 * np.pi * (n * n / (2 * n_chips) + n * (symbol / n_chips - 0.5)
                         - wrapped)
    return np.exp(1j * phase)


def snap_bandwidth(width_hz):
    """The LoRa bandwidth a measured width is close to, or None."""
    best = min(LORA_BANDWIDTHS, key=lambda bw: abs(width_hz - bw))
    return best if abs(width_hz - best) <= SNAP_TOLERANCE * best else None


def symbol_time(sf, bw_hz):
    return 2 ** sf / bw_hz


# --------------------------------------------------------------------------
# Burst detection
# --------------------------------------------------------------------------

def spectrogram(iq, sample_rate, frame=FRAME):
    """(freqs, power[frames, bins]) with bins in increasing frequency."""
    n = len(iq) // frame
    window = np.hanning(frame).astype(np.float32)
    x = np.asarray(iq[:n * frame], dtype=np.complex64).reshape(n, frame)
    spec = sp_fft.fft(x * window, axis=1, workers=-1)
    power = np.fft.fftshift(spec.real ** 2 + spec.imag ** 2, axes=1)
    freqs = np.fft.fftshift(np.fft.fftfreq(frame, 1.0 / sample_rate))
    return freqs, power.astype(np.float32, copy=False)


def box_mean(a, rows, cols):
    """Mean over a rows x cols box centred on each cell (edges clamp).

    Running sums, so it costs the same whatever the box size."""
    out = a.astype(np.float64)
    for axis, size in ((0, rows), (1, cols)):
        if size <= 1:
            continue
        before, after = size // 2, size - 1 - size // 2
        pad = [(0, 0), (0, 0)]
        pad[axis] = (before + 1, after)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        hi = np.take(c, range(size, c.shape[axis]), axis=axis)
        lo = np.take(c, range(0, c.shape[axis] - size), axis=axis)
        out = (hi - lo) / size
    return out


def noise_floor(power):
    """Mean noise power per bin, from a low percentile over time.

    Noise power in one bin is exponentially distributed, whose p-th
    percentile is -ln(1 - p) times its mean; dividing by that gives the
    mean. Smoothed across frequency to steady it.
    """
    p = FLOOR_PERCENTILE / 100.0
    k = int(p * (len(power) - 1))
    low = np.partition(np.ascontiguousarray(power.T), k, axis=1)[:, k]
    floor = low / -np.log(1 - p)
    floor = ndimage.median_filter(floor, size=15, mode="nearest")
    return np.maximum(floor, np.finfo(np.float32).tiny)


def find_bursts(iq, sample_rate=SAMPLE_RATE, threshold_db=DEFAULT_THRESHOLD_DB):
    """Wideband bursts in a block of IQ, in time order.

    Each is a dict: start / stop (sample indices), start_s, duration_s,
    freq_hz (centre, relative to the tuned frequency), width_hz
    (measured at -6 dB), bw_hz (snapped LoRa bandwidth or None), snr_db
    (average over the occupied band while it was on) and cut_off
    ("start", "end", "both" or None: it was already on when the block
    began, or still on when it ended, so duration_s is a lower bound).
    """
    freqs, power = spectrogram(iq, sample_rate)
    n_frames = len(power)
    if n_frames < 8:
        return []
    bin_hz = freqs[1] - freqs[0]
    frame_s = FRAME / sample_rate
    floor = noise_floor(power)
    snr = power / floor
    # Detection runs on a coarse grid, COARSE x COARSE cells averaged.
    c = COARSE
    nc = n_frames // c
    coarse = snr[:nc * c].reshape(nc, c, -1, c).mean(axis=(1, 3))
    box = (max(1, round(BOX_S / frame_s / c)),
           max(1, round(BOX_HZ / bin_hz / c)))
    smooth = box_mean(coarse, *box)
    seeds = smooth > 10 ** (threshold_db / 10)
    grown = smooth > 10 ** (min(GROW_DB, threshold_db) / 10)
    bridge = max(1, round(BRIDGE_S / frame_s))
    labels, _ = ndimage.label(
        ndimage.maximum_filter(grown, size=(max(1, bridge // c), 1)))
    keep = np.zeros(labels.max() + 1, bool)
    keep[np.unique(labels[seeds])] = True
    keep[0] = False
    labels[~keep[labels]] = 0
    bursts = []
    for region in ndimage.find_objects(labels):
        if region is None:
            continue
        rows = slice(region[0].start * c,
                     n_frames if region[0].stop == nc else region[0].stop * c)
        cols = slice(region[1].start * c, region[1].stop * c)
        for a, b in _runs(snr[rows], cols):
            t0, t1 = _on_time(snr, a, b, rows, bridge)
            if t1 is None or (t1 - t0) * frame_s < MIN_BURST_S:
                continue
            excess = snr[t0:t1, a:b + 1].mean() - 1.0
            width = (b - a + 1) * bin_hz
            edge = {(True, False): "start", (False, True): "end",
                    (True, True): "both"}.get((t0 == 0, t1 == n_frames))
            bursts.append({
                "start": t0 * FRAME, "stop": t1 * FRAME,
                "start_s": t0 * frame_s, "duration_s": (t1 - t0) * frame_s,
                "freq_hz": float((freqs[a] + freqs[b]) / 2),
                "width_hz": float(width), "bw_hz": snap_bandwidth(width),
                "snr_db": float(10 * np.log10(max(excess, 1e-3))),
                "cut_off": edge})
    return _merge(bursts, MERGE_GAP_S * sample_rate, snr, freqs, n_frames)


def _overlap(a, b):
    """Hz of spectrum two bursts share."""
    lo = max(a["freq_hz"] - a["width_hz"] / 2, b["freq_hz"] - b["width_hz"] / 2)
    hi = min(a["freq_hz"] + a["width_hz"] / 2, b["freq_hz"] + b["width_hz"] / 2)
    return max(0.0, hi - lo)


def _merge(bursts, gap_samples, snr, freqs, n_frames):
    """Join bursts on the same channel (the narrower mostly inside the
    wider) that overlap or nearly touch in time (a weak packet, or a
    packet's tail, can break off), keeping the wider frequency span
    and re-measuring SNR over the union."""
    bursts.sort(key=lambda b: (b["start"], b["freq_hz"]))
    out = []
    for burst in bursts:
        for prev in out:
            same_channel = _overlap(prev, burst) >= 0.75 * min(
                prev["width_hz"], burst["width_hz"])
            if same_channel and burst["start"] - prev["stop"] < gap_samples:
                if burst["width_hz"] > prev["width_hz"]:
                    for key in ("freq_hz", "width_hz", "bw_hz"):
                        prev[key] = burst[key]
                prev["stop"] = max(prev["stop"], burst["stop"])
                break
        else:
            out.append(dict(burst))
    bin_hz = freqs[1] - freqs[0]
    frame_s = FRAME / (bin_hz * FRAME)
    for b in out:
        t0, t1 = b["start"] // FRAME, b["stop"] // FRAME
        b["duration_s"] = (t1 - t0) * frame_s
        half = b["width_hz"] / 2
        cols = (freqs >= b["freq_hz"] - half) & (freqs <= b["freq_hz"] + half)
        excess = snr[t0:t1][:, cols].mean() - 1.0
        b["snr_db"] = float(10 * np.log10(max(excess, 1e-3)))
        b["cut_off"] = {(True, False): "start", (False, True): "end",
                        (True, True): "both"}.get((t0 == 0, t1 == n_frames))
    return out


def _runs(snr, cols):
    """Occupied bands, as (first bin, last bin), in one active region.

    snr is the region's frames (all bins). Bins whose average excess
    clears the noise by 4 sigma are grouped into runs (gaps up to
    GAP_HZ bridged), so two signals side by side stay separate; each
    run at least MIN_WIDTH_HZ wide is then trimmed to WIDTH_DROP_DB
    below its own peak.
    """
    n_frames, n_bins = snr.shape
    excess = ndimage.uniform_filter1d(snr.mean(axis=0) - 1.0, SPEC_SMOOTH_BINS)
    sigma = 1.0 / np.sqrt(n_frames * SPEC_SMOOTH_BINS)
    pad = cols.stop - cols.start  # look one region-width either side
    lo, hi = max(0, cols.start - pad), min(n_bins, cols.stop + pad)
    above = np.nonzero(excess[lo:hi] > 4 * sigma)[0] + lo
    if len(above) == 0:
        return []
    runs, start, prev = [], above[0], above[0]
    for i in above[1:]:
        if i - prev > GAP_BINS:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    out = []
    for a, b in runs:
        if (b - a + 1) < MIN_WIDTH_BINS:
            continue
        peak = excess[a:b + 1].max()
        keep = np.nonzero(excess[a:b + 1] > peak * 10 ** (-WIDTH_DROP_DB / 10))[0]
        a2, b2 = a + keep[0], a + keep[-1]
        if (b2 - a2 + 1) >= MIN_WIDTH_BINS:
            out.append((int(a2), int(b2)))
    return out


def _on_time(snr, a, b, rows, bridge):
    """(first, stop) frame of a signal in bins a..b, searched a little
    around the region's rows, or (None, None)."""
    n_frames, n_bins = snr.shape
    t_lo = max(0, rows.start - bridge)
    t_hi = min(n_frames, rows.stop + bridge)
    band = snr[t_lo:t_hi, a:b + 1].mean(axis=1) - 1.0
    band = ndimage.uniform_filter1d(band, ON_SMOOTH_FRAMES)
    sigma = 1.0 / np.sqrt((b - a + 1) * ON_SMOOTH_FRAMES)
    core = band[rows.start - t_lo:rows.stop - t_lo]
    level = max(np.median(core) / 2, 5 * sigma)
    join = max(1, round(MERGE_GAP_S * SAMPLE_RATE / FRAME))
    on = ndimage.binary_closing(band > level, np.ones(join, bool))
    on |= band > level  # closing can erode the ends
    runs, n_runs = ndimage.label(on)
    if n_runs == 0:
        return None, None
    peak = rows.start - t_lo + int(np.argmax(core))
    run = runs[peak] or runs[np.nonzero(runs)[0][
        np.argmin(np.abs(np.nonzero(runs)[0] - peak))]]
    idx = np.nonzero(runs == run)[0]
    return t_lo + idx[0], t_lo + idx[-1] + 1


# --------------------------------------------------------------------------
# Chirp analysis
# --------------------------------------------------------------------------

def to_baseband(iq, sample_rate, freq_hz, bw_hz):
    """Move a signal at freq_hz to 0 Hz and resample to bw_hz (one sample
    per chip), filtering out everything outside the channel."""
    t = np.arange(len(iq)) / sample_rate
    shifted = np.asarray(iq) * np.exp(-2j * np.pi * freq_hz * t)
    ratio = Fraction(int(bw_hz), int(sample_rate)).limit_denominator(1000)
    return resample_poly(shifted, ratio.numerator, ratio.denominator)


def dechirp_score(chips, sf, direction="up", windows=WINDOWS):
    """How strongly `chips` (baseband, one sample per chip) looks like a
    run of identical chirps of this SF, in dB; None if too short.

    Sums the power spectra of up to `windows` dechirped symbols, starting
    half a symbol in, and returns the strongest pair of neighbouring bins
    over the average pair. Noise scores about 4 dB; a clean preamble
    scores 14 dB or more.
    """
    n = 2 ** sf
    k = min(windows, (len(chips) - n // 2) // n)
    if k < MIN_WINDOWS:
        return None
    segs = np.asarray(chips[n // 2:n // 2 + k * n]).reshape(k, n)
    ref = upchirp(sf)
    ref = np.conj(ref) if direction == "up" else ref
    spec = (np.abs(np.fft.fft(segs * ref, axis=1)) ** 2).sum(axis=0)
    pair = spec + np.roll(spec, -1)
    return float(10 * np.log10(pair.max() / (2 * spec.mean())))


def chirp_facts(iq, sample_rate, freq_hz, bw_hz, sfs=SPREADING_FACTORS):
    """Spreading factor and chirp direction of a burst.

    iq should start at (or just before) the burst. Returns a dict with
    sf, chirp ("up" or "down"), score_db, margin_db (over the runner-up)
    and is_lora; sf and chirp are None when nothing scores
    CHIRP_THRESHOLD_DB.
    """
    # Enough samples for WINDOWS + 1 symbols of the slowest SF tried.
    need = int((WINDOWS + 1) * symbol_time(max(sfs), bw_hz) * sample_rate)
    chips = to_baseband(iq[:need], sample_rate, freq_hz, bw_hz)
    scores = []
    for sf in sfs:
        for direction in ("up", "down"):
            score = dechirp_score(chips, sf, direction)
            if score is not None:
                scores.append((score, sf, direction))
    scores.sort(reverse=True)
    if not scores or scores[0][0] < CHIRP_THRESHOLD_DB:
        best = scores[0][0] if scores else None
        return {"sf": None, "chirp": None, "score_db": best,
                "margin_db": None, "is_lora": False}
    score, sf, direction = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    return {"sf": sf, "chirp": direction, "score_db": score,
            "margin_db": score - runner_up, "is_lora": True}


def analyse(iq, sample_rate=SAMPLE_RATE, threshold_db=DEFAULT_THRESHOLD_DB):
    """find_bursts() plus chirp_facts() for each: the full fact sheet.

    A burst counts as LoRa only if its width snaps to a LoRa bandwidth,
    it dechirps cleanly at some SF, and it lasts at least
    MIN_PACKET_SYMBOLS symbols at that SF (or was cut off by the edge of
    the block). Adds sf, chirp, chirp_score_db, is_lora, symbol_s and
    symbols.
    """
    out = []
    for burst in find_bursts(iq, sample_rate, threshold_db):
        burst.update(sf=None, chirp=None, chirp_score_db=None,
                     is_lora=False, symbol_s=None, symbols=None)
        if burst["bw_hz"]:
            start = max(0, burst["start"] - FRAME)
            facts = chirp_facts(iq[start:burst["stop"]], sample_rate,
                                burst["freq_hz"], burst["bw_hz"])
            burst["chirp_score_db"] = facts["score_db"]
            if facts["is_lora"]:
                symbol_s = symbol_time(facts["sf"], burst["bw_hz"])
                symbols = burst["duration_s"] / symbol_s
                if symbols >= MIN_PACKET_SYMBOLS or burst["cut_off"]:
                    burst.update(sf=facts["sf"], chirp=facts["chirp"],
                                 is_lora=True, symbol_s=symbol_s,
                                 symbols=symbols)
        out.append(burst)
    return out
