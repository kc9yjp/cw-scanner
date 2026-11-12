#!/usr/bin/env python3
"""cwscan.py — a simple CW (Morse code) scanner and listener for RTL-SDR.

Two modes:

    python3 cwscan.py scan
        Sweep a band segment (default: the 40 m CW segment, 7.000-7.040 MHz)
        and print the frequencies where signals are found.

    python3 cwscan.py listen 7030000
        Tune to one frequency, save what is heard as a WAV file, and print a
        live Morse decode of it.

How scanning works
------------------
An RTL-SDR dongle can only see a slice of spectrum at a time (here 250 kHz
by default, sampled at 250 kS/s). So `scan` hops across the requested range
in overlapping chunks. For each chunk it:

1. Captures IQ samples and averages several FFTs into one power spectrum.
   Averaging knocks the noise down while real signals stay put.
2. Estimates the noise floor with a rolling median across the spectrum.
   A median is blind to the narrow spikes that signals make, so it tracks
   the noise even when signals are present. This is the same idea as
   MultiScan SDR's "pseudo-spectrum": what matters is how far a signal
   rises above its *local* noise, not its absolute power.
3. Flags every group of bins that rises more than --threshold dB above the
   local floor, and reports the strongest bin of each group.

How listening works
-------------------
The dongle is tuned to the requested frequency, so the CW signal sits near
0 Hz in the baseband. The signal is:

1. Low-pass filtered to --cutoff Hz (default 400 Hz). This is the narrow
   CW filter: it keeps the one signal and throws away everything else in
   the 250 kHz the dongle sees.
2. Mixed up by --tone Hz (default 700 Hz) so the carrier becomes an audible
   tone, then written to a WAV file.
3. The magnitude of the filtered signal is its keying envelope (key down =
   loud, key up = quiet). That envelope is thresholded into a 0/1 waveform
   and fed to the Morse decoder in morse.py.

Hardware note: the dongle code is the only part that needs pyrtlsdr and
librtlsdr. Everything else (FFT, filtering, decoding) is pure numpy/scipy
and is exercised by test_synth.py without any hardware attached.
"""

import argparse
import sys
import time
import wave

import numpy as np
from scipy import ndimage, signal

import morse

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

DEFAULT_SAMPLE_RATE = 250_000   # Hz. Narrow on purpose: at 250 kS/s a 4096
                                # point FFT gives ~61 Hz bins, fine for CW
                                # signals that are only 100-500 Hz wide.
DEFAULT_FFT_SIZE = 4096
HOP_OVERLAP = 0.2               # neighbouring chunks overlap by 20% so a
                                # signal on a chunk edge is still caught.
DEFAULT_THRESHOLD_DB = 10.0     # how far above the local floor a signal
                                # must rise to be reported.
DEFAULT_CUTOFF_HZ = 400.0       # width of the narrow CW filter.
DEFAULT_TONE_HZ = 700.0         # pitch of the audio tone in listen mode.
AUDIO_RATE = 8000               # Hz, sample rate of the saved WAV file.


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def parse_freq(text):
    """Parse '7030000', '7.03M', '7.03MHz', '7030k' or '7030kHz' -> Hz."""
    s = text.strip().lower()
    for suffix, mult in (("mhz", 1e6), ("m", 1e6), ("khz", 1e3), ("k", 1e3)):
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult
    return float(s)


def fmt_freq(hz):
    """7030200 -> '7.0302 MHz'."""
    if hz >= 1e6:
        return "%.4f MHz" % (hz / 1e6)
    if hz >= 1e3:
        return "%.1f kHz" % (hz / 1e3)
    return "%.0f Hz" % hz


# --------------------------------------------------------------------------
# DSP: spectrum, detection, demodulation (no hardware needed)
# --------------------------------------------------------------------------

def averaged_psd(iq, sample_rate, fft_size=DEFAULT_FFT_SIZE):
    """Average power spectrum of IQ samples.

    Returns (freqs_hz, psd_db) where freqs are relative to the tuner centre
    frequency, from -sample_rate/2 to +sample_rate/2. Several Hann-windowed
    FFTs are averaged, which smooths the noise without touching signals.
    """
    iq = np.asarray(iq)
    window = np.hanning(fft_size)
    n_seg = len(iq) // fft_size
    if n_seg < 1:
        raise ValueError("not enough samples for one FFT (%d needed)" % fft_size)
    blocks = iq[:n_seg * fft_size].reshape(n_seg, fft_size)
    spec = np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1)
    psd = np.mean(np.abs(spec) ** 2, axis=0)
    psd_db = 10 * np.log10(psd + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0)) * sample_rate
    return freqs, psd_db


def noise_floor_db(psd_db, median_size=151):
    """Rolling-median noise floor.

    The window (151 bins ~ 9 kHz at the default settings) is far wider than
    any CW signal, so signals barely move the median while the noise floor
    -- which can slope across the band -- is tracked faithfully.
    """
    size = min(median_size, len(psd_db))
    if size % 2 == 0:
        size -= 1
    if size < 3:
        return np.full_like(psd_db, np.median(psd_db))
    return ndimage.median_filter(psd_db, size=size, mode="nearest")


def find_candidates(freqs_hz, psd_db, threshold_db=DEFAULT_THRESHOLD_DB):
    """Group bins that rise above the local noise floor into candidates.

    Returns a list of dicts: relative frequency, peak level, and strength
    above the local floor, strongest first.
    """
    floor = noise_floor_db(psd_db)
    above = (psd_db - floor) > threshold_db
    candidates = []
    i, n = 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            k = i + int(np.argmax(psd_db[i:j]))
            candidates.append({
                "freq_hz": float(freqs_hz[k]),
                "peak_db": float(psd_db[k]),
                "snr_db": float(psd_db[k] - floor[k]),
            })
            i = j
        else:
            i += 1
    candidates.sort(key=lambda c: c["snr_db"], reverse=True)
    return candidates


def detect_signals(iq, sample_rate, fft_size=DEFAULT_FFT_SIZE,
                   threshold_db=DEFAULT_THRESHOLD_DB):
    """Full detection chain for one chunk of IQ: spectrum -> candidates."""
    freqs, psd = averaged_psd(iq, sample_rate, fft_size)
    return find_candidates(freqs, psd, threshold_db)


def demodulate_cw(iq, sample_rate, tone_hz=DEFAULT_TONE_HZ,
                  cutoff_hz=DEFAULT_CUTOFF_HZ, audio_rate=AUDIO_RATE):
    """Demodulate a CW signal sitting near 0 Hz in the baseband.

    Returns (audio, envelope, audio_rate):
      * audio    -- the signal as an audible tone, resampled to audio_rate.
      * envelope -- the keying envelope (magnitude of the filtered signal),
                    resampled to audio_rate, scaled 0..1. Key down is ~1,
                    key up is ~0. This is what the Morse decoder eats.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    # Narrow CW filter around the carrier (zero-phase so the envelope stays
    # time-aligned with the audio).
    sos = signal.butter(4, cutoff_hz, btype="lowpass",
                        fs=sample_rate, output="sos")
    filt = signal.sosfiltfilt(sos, iq)
    # Shift the carrier up to an audible tone and keep the real part.
    t = np.arange(len(iq)) / sample_rate
    audio = np.real(filt * np.exp(-2j * np.pi * tone_hz * t))
    # Remove the double-frequency image left by taking the real part.
    sos_audio = signal.butter(4, 1200, btype="lowpass",
                              fs=sample_rate, output="sos")
    audio = signal.sosfiltfilt(sos_audio, audio)
    # Envelope: magnitude of the filtered signal, lightly smoothed so that
    # individual RF cycles don't chop the keying up.
    env = np.abs(filt)
    smooth_n = max(1, int(0.005 * sample_rate))  # ~5 ms moving average
    env = np.convolve(env, np.ones(smooth_n) / smooth_n, mode="same")
    # Resample both down to audio rate (250 kS/s -> 8 kS/s).
    audio_rs = signal.resample_poly(audio, audio_rate, sample_rate)
    env_rs = signal.resample_poly(env, audio_rate, sample_rate)
    peak = np.max(env_rs) if len(env_rs) else 0.0
    if peak > 0:
        env_rs = env_rs / peak
    return audio_rs, env_rs, audio_rate


def envelope_to_binary(env, sample_rate, frac=0.3, min_ms=15):
    """Threshold a 0..1 envelope into a clean 0/1 keying waveform.

    The threshold sits at `frac` of the 99th percentile, which adapts to
    the signal level. Morphological opening/closing then removes glitches
    shorter than `min_ms` -- a dit at 20 WPM is 60 ms, so 15 ms is safely
    below any real element while killing noise spikes.
    """
    env = np.asarray(env, dtype=float)
    if len(env) == 0:
        return env
    peak = np.percentile(env, 99)
    if peak <= 0:
        return np.zeros_like(env)
    binary = env > frac * peak
    n = max(1, int(min_ms / 1000.0 * sample_rate))
    structure = np.ones(n, dtype=bool)
    binary = ndimage.binary_opening(binary, structure=structure)
    binary = ndimage.binary_closing(binary, structure=structure)
    return binary.astype(float)


def write_wav(path, audio, sample_rate):
    """Write mono 16-bit PCM WAV, normalising to full scale."""
    audio = np.asarray(audio, dtype=float)
    peak = np.max(np.abs(audio)) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


# --------------------------------------------------------------------------
# Hardware: the only part that touches the dongle
# --------------------------------------------------------------------------

def open_sdr(args):
    """Open the RTL-SDR dongle and apply the common settings.

    Raises SystemExit with a plain-English message if pyrtlsdr/librtlsdr
    is missing or no dongle is found.
    """
    try:
        from rtlsdr import RtlSdr
    except ImportError:
        raise SystemExit(
            "pyrtlsdr is not installed (or its C library librtlsdr is "
            "missing).\nInstall it with:\n"
            "  sudo apt install rtl-sdr librtlsdr-dev\n"
            "  pip install --break-system-packages pyrtlsdr numpy scipy\n"
            "or use a virtualenv. See README.md.")
    try:
        sdr = RtlSdr()
    except Exception as exc:  # noqa: BLE001 - hardware errors vary
        raise SystemExit(
            "Could not open an RTL-SDR dongle: %s\n"
            "Is one plugged in, and does your user have USB permission "
            "(udev rules / plugdev group)?" % exc)
    sdr.sample_rate = args.sample_rate
    sdr.gain = "auto" if args.gain == "auto" else float(args.gain)
    if args.ppm:
        sdr.freq_correction = args.ppm
    if args.direct_sampling != "off":
        # V3 direct sampling: 1 = I-ADC, 2 = Q-ADC. Q-ADC is the usual
        # choice for HF. Only meaningful below ~30 MHz.
        sdr.direct_sampling = {"i": 1, "q": 2}[args.direct_sampling]
    return sdr


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def cmd_scan(args):
    start = parse_freq(args.start)
    stop = parse_freq(args.stop)
    if stop <= start:
        raise SystemExit("--stop must be above --start.")
    sdr = open_sdr(args)
    hop = args.sample_rate * (1.0 - HOP_OVERLAP)
    print("Scanning %s to %s in %s chunks at %d S/s ..."
          % (fmt_freq(start), fmt_freq(stop), fmt_freq(hop),
             args.sample_rate))
    found = []
    try:
        centre = start + args.sample_rate / 2.0
        while centre - args.sample_rate / 2.0 < stop:
            sdr.center_freq = centre
            time.sleep(0.1)  # let the tuner and AGC settle
            iq = sdr.read_samples(65536)
            for cand in detect_signals(iq, args.sample_rate,
                                       args.fft_size, args.threshold):
                absolute = centre + cand["freq_hz"]
                if start <= absolute <= stop:
                    found.append((absolute, cand["snr_db"]))
                    print("  %s  +%5.1f dB above floor"
                          % (fmt_freq(absolute), cand["snr_db"]))
            centre += hop
    finally:
        sdr.close()
    found.sort(key=lambda item: item[1], reverse=True)
    print()
    if found:
        print("%d signal(s) found, strongest first:" % len(found))
        for freq, snr in found:
            print("  %s  +%5.1f dB" % (fmt_freq(freq), snr))
    else:
        print("No signals above %.0f dB in %s-%s."
              % (args.threshold, fmt_freq(start), fmt_freq(stop)))


def cmd_listen(args):
    freq = parse_freq(args.freq)
    sdr = open_sdr(args)
    sdr.center_freq = freq
    time.sleep(0.2)  # let the tuner and AGC settle
    out_path = args.output or ("cw-%.0f.wav" % freq)
    print("Listening on %s for %.0f s (tone %d Hz, filter %d Hz) ..."
          % (fmt_freq(freq), args.duration, args.tone, args.cutoff))
    print("Decoded text will appear below; Ctrl-C stops early.")
    chunks = []
    text = ""
    try:
        chunk_samples = int(args.sample_rate)  # 1 second per chunk
        deadline = time.time() + args.duration
        while time.time() < deadline:
            chunks.append(sdr.read_samples(chunk_samples))
            joined = np.concatenate(chunks)
            audio, env, audio_rate = demodulate_cw(
                joined, args.sample_rate,
                tone_hz=args.tone, cutoff_hz=args.cutoff)
            binary = envelope_to_binary(env, audio_rate)
            text = morse.decode_envelope(binary, audio_rate, wpm=args.wpm)
            print("\r%s" % text, end="", flush=True)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        sdr.close()
    print()
    if chunks:
        joined = np.concatenate(chunks)
        audio, _, audio_rate = demodulate_cw(
            joined, args.sample_rate,
            tone_hz=args.tone, cutoff_hz=args.cutoff)
        write_wav(out_path, audio, audio_rate)
        print("Saved %s (%.1f s of audio at %d Hz)."
              % (out_path, len(audio) / audio_rate, audio_rate))
    print("Decoded: %s" % (text or "(nothing decoded)"))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="cwscan.py",
        description="Simple CW (Morse) scanner and listener for RTL-SDR.")
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p):
        p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE,
                       help="dongle sample rate in S/s (default %(default)s)")
        p.add_argument("--gain", default="auto",
                       help="'auto' or a gain in dB (default %(default)s)")
        p.add_argument("--ppm", type=int, default=0,
                       help="frequency correction in ppm (default %(default)s)")
        p.add_argument("--direct-sampling", choices=["off", "i", "q"],
                       default="off",
                       help="V3 HF direct sampling: off, i or q "
                            "(default %(default)s). V4 owners don't need this.")

    p_scan = sub.add_parser("scan", help="sweep a band and list signals")
    p_scan.add_argument("--start", default="7.0M",
                        help="segment start (default %(default)s)")
    p_scan.add_argument("--stop", default="7.04M",
                        help="segment end (default %(default)s)")
    p_scan.add_argument("--threshold", type=float,
                        default=DEFAULT_THRESHOLD_DB,
                        help="dB above the local floor to report "
                             "(default %(default)s)")
    p_scan.add_argument("--fft-size", type=int, default=DEFAULT_FFT_SIZE,
                        help="FFT size (default %(default)s)")
    add_common(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_listen = sub.add_parser("listen", help="tune, record, and decode one "
                                             "frequency")
    p_listen.add_argument("freq", help="frequency, e.g. 7030000 or 7.03M")
    p_listen.add_argument("--duration", type=float, default=10,
                          help="seconds to listen (default %(default)s)")
    p_listen.add_argument("--output", default=None,
                          help="WAV file to write (default cw-<freq>.wav)")
    p_listen.add_argument("--tone", type=float, default=DEFAULT_TONE_HZ,
                          help="audio tone pitch in Hz (default %(default)s)")
    p_listen.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF_HZ,
                          help="CW filter width in Hz (default %(default)s)")
    p_listen.add_argument("--wpm", type=float, default=None,
                          help="Morse speed to decode at; if omitted the "
                               "decoder estimates it from the signal")
    add_common(p_listen)
    p_listen.set_defaults(func=cmd_listen)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
