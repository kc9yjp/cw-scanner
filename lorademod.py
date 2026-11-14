"""lorademod.py — LoRa physical layer: decode (and encode) packets.

Hardware-free, like lora.py. decode() takes the samples of one packet
found by lora.analyse() and returns its header, payload bytes and
whether the payload CRC checks out. encode() and modulate() build test
packets (used by the self-test, and handy for experiments).

The bit-level chain follows the open-source gr-lora_sdr receiver (EPFL,
Tapparel et al.), which matches Semtech SX127x/SX126x radios:

  payload bytes -> whitening -> nibbles (low nibble first)
  [explicit header: length, coding rate, CRC flag, 5-bit checksum]
  -> Hamming(4, 4+CR) codewords -> diagonal interleaving
  -> Gray demapping -> +1 -> chirp symbols

The first 8 symbols (header block) always use CR 4/8 and carry SF-2 bits
each ("reduced rate"); payload blocks carry SF bits per symbol (SF-2
with low data rate optimisation, LDRO, which radios switch on when a
symbol lasts more than 16 ms).

Synchronisation
---------------
A dechirped preamble up-chirp peaks at bin k_up = CFO + tau, and an SFD
down-chirp at k_down = CFO - tau (tau: how far our symbol grid is from
the packet's, in chips; CFO in bins), so the two give both, up to an
N/2 ambiguity; of the candidate alignments, the one on which the header
and SFD dechirp most sharply wins. These estimates are made at 8
samples per chip, where they are unbiased. After that the timing is
tracked symbol by symbol, which follows the slow drift between the two
radios' clocks over a long packet.
"""

import numpy as np
from scipy.signal import firwin, resample_poly
from fractions import Fraction

import lora

OS = 8                          # oversampling while aligning: 1/8 chip.
TRACK_PAD = 2                   # FFT zero-padding when tracking timing.
TRACK_GAIN = 0.3                # share of each symbol's timing error fixed.
MAX_PREAMBLE = 64               # symbols searched for the preamble.
LDRO_SYMBOL_S = 0.016           # LDRO on above this symbol time.

SYNC_WORDS = {0x12: "private", 0x34: "LoRaWAN", 0x2B: "Meshtastic"}


# --------------------------------------------------------------------------
# Bit-level building blocks
# --------------------------------------------------------------------------

def _whitening_sequence(n=256):
    """LoRa's whitening bytes: LFSR x^8 + x^6 + x^5 + x^4 + 1 from 0xFF."""
    out, state = [], 0xFF
    for _ in range(n):
        out.append(state)
        bit = ((state >> 7) ^ (state >> 5) ^ (state >> 4) ^ (state >> 3)) & 1
        state = ((state << 1) | bit) & 0xFF
    return bytes(out)


WHITENING = _whitening_sequence()


def whiten(data):
    return bytes(b ^ WHITENING[i] for i, b in enumerate(data))


def crc16(data):
    """CRC-16/CCITT (poly 0x1021, init 0), bit by bit, MSB first."""
    crc = 0
    for byte in data:
        for _ in range(8):
            if ((crc >> 8) ^ byte) & 0x80:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
            byte = (byte << 1) & 0xFF
    return crc


def payload_crc(payload):
    """The 16-bit CRC a LoRa radio sends after the payload: the CRC of all
    but the last two bytes, XORed with those two bytes."""
    payload = bytes(payload)
    if len(payload) < 2:
        return crc16(payload)
    return crc16(payload[:-2]) ^ payload[-1] ^ (payload[-2] << 8)


def header_checksum(n0, n1, n2):
    """5-bit checksum of the explicit header's three nibbles."""
    b = [[(n >> i) & 1 for i in (3, 2, 1, 0)] for n in (n0, n1, n2)]
    (a3, a2, a1, a0), (b3, b2, b1, b0), (c3, c2, c1, c0) = b
    x4 = a3 ^ a2 ^ a1 ^ a0
    x3 = a3 ^ b3 ^ b2 ^ b1 ^ c0
    x2 = a2 ^ b3 ^ b0 ^ c3 ^ c1
    x1 = a1 ^ b2 ^ b0 ^ c2 ^ c1 ^ c0
    x0 = a0 ^ b1 ^ c3 ^ c2 ^ c1 ^ c0
    return (x4 << 4) | (x3 << 3) | (x2 << 2) | (x1 << 1) | x0


def hamming_encode(nibble, cr):
    """4 data bits -> (4 + cr)-bit codeword: data LSB first, then parity."""
    d3, d2, d1, d0 = [(nibble >> i) & 1 for i in (3, 2, 1, 0)]
    if cr == 1:
        return (d0 << 4) | (d1 << 3) | (d2 << 2) | (d3 << 1) | (d0 ^ d1 ^ d2 ^ d3)
    p0 = d0 ^ d1 ^ d2
    p1 = d1 ^ d2 ^ d3
    p2 = d0 ^ d1 ^ d3
    p3 = d0 ^ d2 ^ d3
    cw = (d0 << 7) | (d1 << 6) | (d2 << 5) | (d3 << 4) | (p0 << 3) | (p1 << 2) \
        | (p2 << 1) | p3
    return cw >> (4 - cr)


def hamming_decode(codeword, cr):
    """(4 + cr)-bit codeword -> (nibble, fixed): CR 4/7 and 4/8 correct one
    bit error, 4/5 and 4/6 only carry it through."""
    n = 4 + cr
    c = [(codeword >> (n - 1 - i)) & 1 for i in range(n)]   # MSB first
    data = [c[3], c[2], c[1], c[0]]                          # d3 d2 d1 d0
    fixed = False
    if cr >= 3 and not (cr == 4 and sum(c) % 2 == 0):
        s0 = c[0] ^ c[1] ^ c[2] ^ c[4]
        s1 = c[1] ^ c[2] ^ c[3] ^ c[5]
        s2 = c[0] ^ c[1] ^ c[3] ^ c[6]
        flip = {5: 3, 7: 2, 3: 1, 6: 0}.get(s0 | (s1 << 1) | (s2 << 2))
        if flip is not None:
            data[flip] ^= 1
            fixed = True
    return (data[0] << 3) | (data[1] << 2) | (data[2] << 1) | data[3], fixed


def gray(x):
    return x ^ (x >> 1)


def gray_inverse(x, bits):
    out = x
    for j in range(1, bits):
        out ^= x >> j
    return out


def interleave(codewords, sf_app, cw_len, reduced):
    """sf_app codewords -> cw_len symbol values (SF bits each when
    reduced: a parity bit and a zero are appended)."""
    bits = [[(cw >> (cw_len - 1 - k)) & 1 for k in range(cw_len)]
            for cw in codewords]
    out = []
    for i in range(cw_len):
        row = [bits[(i - j - 1) % sf_app][i] for j in range(sf_app)]
        if reduced:
            row += [sum(row) % 2, 0]
        value = 0
        for bit in row:
            value = (value << 1) | bit
        out.append(value)
    return out


def deinterleave(values, sf_app, cw_len):
    """cw_len symbol values (sf_app bits each) -> sf_app codewords."""
    rows = [[(v >> (sf_app - 1 - j)) & 1 for j in range(sf_app)]
            for v in values]
    cws = [[0] * cw_len for _ in range(sf_app)]
    for i in range(cw_len):
        for j in range(sf_app):
            cws[(i - j - 1) % sf_app][i] = rows[i][j]
    out = []
    for cw in cws:
        value = 0
        for bit in cw:
            value = (value << 1) | bit
        out.append(value)
    return out


def use_ldro(sf, bw_hz):
    return lora.symbol_time(sf, bw_hz) > LDRO_SYMBOL_S


# --------------------------------------------------------------------------
# Encoder (for tests)
# --------------------------------------------------------------------------

def encode(payload, sf, cr=1, has_crc=True, ldro=False):
    """Payload bytes -> data symbol values (explicit header)."""
    payload = bytes(payload)
    nib = [len(payload) >> 4, len(payload) & 0xF, (cr << 1) | int(has_crc)]
    chk = header_checksum(*nib)
    nib += [chk >> 4, chk & 0xF]
    for b in whiten(payload):
        nib += [b & 0xF, b >> 4]
    if has_crc:
        crc = payload_crc(payload)
        nib += [crc & 0xF, (crc >> 4) & 0xF, (crc >> 8) & 0xF, crc >> 12]
    n = 2 ** sf
    symbols, pos, first = [], 0, True
    while pos < len(nib):
        reduced = first or ldro
        sf_app = sf - 2 if reduced else sf
        cw_cr = 4 if first else cr
        block = nib[pos:pos + sf_app]
        block += [0] * (sf_app - len(block))
        pos += sf_app
        cws = [hamming_encode(x, cw_cr) for x in block]
        for v in interleave(cws, sf_app, 4 + cw_cr, reduced):
            symbols.append((gray_inverse(v, sf) + 1) % n)
        first = False
    return symbols


def modulate(data_symbols, sf, bw_hz, sample_rate, sync_word=0x12,
             preamble=8, offset_hz=0.0, invert=False):
    """Chirps for a whole packet at sample_rate: preamble, sync word,
    2.25 down-chirps, data."""
    n = 2 ** sf
    seq = ([0] * preamble + [(sync_word >> 4) * 8, (sync_word & 0xF) * 8]
           + ["down", "down", "quarter"] + list(data_symbols))
    lengths = np.array([n // 4 if s == "quarter" else n for s in seq])
    starts = np.concatenate([[0], np.cumsum(lengths)])
    # One continuous timeline, in chips, so symbols don't drift.
    t = np.arange(int(starts[-1] * sample_rate / bw_hz)) * bw_hz / sample_rate
    seg = np.searchsorted(starts, t, side="right") - 1
    k = t - starts[seg]
    down = np.array([s in ("down", "quarter") for s in seq])[seg]
    value = np.array([0 if isinstance(s, str) else s for s in seq])[seg]
    freq = np.where(down, bw_hz * (0.5 - k / n),
                    bw_hz * (((k + value) % n) / n - 0.5))
    if invert:
        freq = -freq
    return np.exp(1j * np.cumsum(2 * np.pi * (freq + offset_hz) / sample_rate))


# --------------------------------------------------------------------------
# Decoder
# --------------------------------------------------------------------------

def _oversampled(iq, sample_rate, freq_hz, bw_hz):
    """Packet at 0 Hz, OS samples per chip, filtered to the channel."""
    t = np.arange(len(iq)) / sample_rate
    x = np.asarray(iq) * np.exp(-2j * np.pi * freq_hz * t)
    ratio = Fraction(int(OS * bw_hz), int(sample_rate)).limit_denominator(1000)
    x = resample_poly(x, ratio.numerator, ratio.denominator)
    # Low-pass to just over the channel edges (+-0.55 BW); firwin's cutoff
    # is a fraction of Nyquist, which is OS * BW / 2 here.
    taps = firwin(8 * OS + 1, 0.55 * 2 / OS)
    return np.convolve(x, taps, mode="same")


def _peak(window, ref, pad=8):
    """Fractional FFT bin of the strongest tone after dechirping, and how
    far it stands above the average bin."""
    spec = np.abs(np.fft.fft(window * ref, pad * len(window)))
    i = int(np.argmax(spec))
    return i / pad, spec[i] / (np.mean(spec) + 1e-12)


def _peak_os(power, n, pad):
    """Bin (mod n, fractional) of the strongest tone in an oversampled
    dechirped power spectrum of pad * n * OS points.

    At one sample per chip a symbol that is tau chips off splits into
    two pieces whose phase jump pulls the peak toward a whole bin; at OS
    samples per chip the pieces are separate clean tones (at k and k-n),
    so the estimate is unbiased."""
    i = int(np.argmax(power))
    f = i / pad
    if f > n * OS / 2:
        f -= n * OS
    return f % n


class _Aligned:
    """Samples one symbol at a time on a given alignment."""

    def __init__(self, x, n, cfo_bins, t0_chips, up):
        self.n, self.t0, self.up = n, t0_chips, up
        t = np.arange(len(x)) / OS          # in chips
        self.x = x * np.exp(-2j * np.pi * cfo_bins / n * t)
        self.adj = 0.0                      # timing tracking, in chips
        self.up_os = np.conj(lora.upchirp(int(np.log2(n)), samples_per_chip=OS))

    def window_at(self, t_chips):
        """One symbol's chips starting t_chips into the samples, or None
        if that runs off either end."""
        pos = (t_chips + np.arange(self.n)) * OS
        idx = np.round(pos).astype(int)
        if idx[0] < 0 or idx[-1] >= len(self.x):
            return None
        return self.x[idx]

    def value_at(self, t_chips):
        w = self.window_at(t_chips)
        if w is None:
            return None
        return int(np.argmax(np.abs(np.fft.fft(w * np.conj(self.up)))))

    def value(self, k):
        """Symbol k counting from t0."""
        return self.value_at(self.t0 + k * self.n)

    def tracked(self, k):
        """Symbol k, nudging the timing after each one so that its peak
        stays centred on a bin. That follows the drift from the two
        radios' clocks differing (several ppm: over a 600 ms packet the
        symbols slide a chip or more)."""
        t = self.t0 + k * self.n + self.adj
        w = self.window_at(t)
        if w is None:
            return None
        # The value from one sample per chip, where a symbol's two pieces
        # fold together (best against noise) ...
        v = int(np.argmax(np.abs(np.fft.fft(w * np.conj(self.up)))))
        # ... and the timing error from OS samples per chip, where the
        # fractional peak isn't biased. Late by e chips shows as +e.
        i0 = int(round(t * OS))
        w_os = self.x[i0:i0 + self.n * OS]
        if len(w_os) == self.n * OS:
            power = np.abs(np.fft.fft(w_os * self.up_os,
                                      TRACK_PAD * self.n * OS)) ** 2
            err = (_peak_os(power, self.n, TRACK_PAD) - v + self.n / 2) \
                % self.n - self.n / 2
            if abs(err) < 0.75:         # ignore a symbol read wrongly
                self.adj -= TRACK_GAIN * err
        return v

    def sharpness(self, starts, down=False):
        """Mean share of each window's energy in its strongest bin (windows
        starting at these chip positions, dechirped as up- or down-chirps):
        near 1 when aligned, about half when off by half a symbol."""
        ref = self.up if down else np.conj(self.up)
        total = 0.0
        for t in starts:
            w = self.window_at(t)
            if w is None:
                return None
            p = np.abs(np.fft.fft(w * ref)) ** 2
            total += p.max() / p.sum()
        return total / len(starts)


def _demod_block(values, sf, cw_cr, reduced, offset):
    n = 2 ** sf
    sf_app = sf - 2 if reduced else sf
    out = []
    for v in values:
        s = (v - offset) % n
        if reduced:
            s >>= 2
        out.append(gray(s))
    nibbles, fixed = [], 0
    for cw in deinterleave(out, sf_app, 4 + cw_cr):
        nib, f = hamming_decode(cw, cw_cr)
        nibbles.append(nib)
        fixed += f
    return nibbles, fixed


def _header(a, sf, offset):
    values = [a.tracked(k) for k in range(8)]
    if None in values:
        return None
    nib, fixed = _demod_block(values, sf, 4, True, offset)
    length = (nib[0] << 4) | nib[1]
    cr, has_crc = nib[2] >> 1, nib[2] & 1
    chk = ((nib[3] & 1) << 4) | nib[4]
    ok = chk == header_checksum(nib[0], nib[1], nib[2]) and 1 <= cr <= 4 \
        and length > 0
    return {"ok": ok, "length": length, "cr": cr, "has_crc": bool(has_crc),
            "extra": nib[5:], "fixed": fixed}


def decode(iq, sample_rate, freq_hz, bw_hz, sf, inverted=False, ldro=None):
    """Decode one LoRa packet.

    iq holds the packet (from a little before it starts) and freq_hz is
    its offset from the tuned frequency. Returns a
    dict: ok (payload CRC good, or header good when there's no CRC),
    error (why not, or None), sync_word, length, cr, has_crc, ldro,
    payload (bytes), crc_ok, cfo_hz, fixed (bit errors corrected).
    """
    n = 2 ** sf
    ldro = use_ldro(sf, bw_hz) if ldro is None else ldro
    result = {"ok": False, "error": None, "sync_word": None, "length": None,
              "cr": None, "has_crc": None, "ldro": ldro, "payload": b"",
              "crc_ok": None, "cfo_hz": None, "fixed": 0}
    x = _oversampled(iq, sample_rate, freq_hz, bw_hz)
    if inverted:
        x = np.conj(x)
    up = lora.upchirp(sf)
    step = n * OS
    n_win = min(len(x) // step, MAX_PREAMBLE + 8)
    wins = [x[i * step:(i + 1) * step:OS] for i in range(n_win)]
    ups = [_peak(w, np.conj(up), pad=1) for w in wins]

    # Preamble: the first run of 4+ windows peaking at the same bin.
    run_start, run_end = None, None
    i = 0
    while i < n_win:
        j = i
        while j + 1 < n_win and min((ups[j + 1][0] - ups[i][0]) % n,
                                    (ups[i][0] - ups[j + 1][0]) % n) <= 1:
            j += 1
        if j - i + 1 >= 4:
            run_start, run_end = i, j
            break
        i += 1
    if run_start is None:
        result["error"] = "no preamble"
        return result
    up_os = lora.upchirp(sf, samples_per_chip=OS)
    pad = 4

    def os_power(k, ref):
        w = x[k * step:(k + 1) * step]
        return np.abs(np.fft.fft(w * ref, pad * step)) ** 2

    inner = range(run_start + 1, max(run_start + 2, run_end))
    k_up = _peak_os(sum(os_power(k, np.conj(up_os)) for k in inner), n, pad)
    # SFD: the window just after the preamble that is most down-chirp-like.
    best, sfd_win = 0.0, None
    for k in range(run_end + 1, min(run_end + 6, n_win)):
        score = _peak(wins[k], up, pad=1)[1]
        if score > best:
            best, sfd_win = score, k
    if sfd_win is None:
        result["error"] = "no SFD"
        return result
    k_down = _peak_os(os_power(sfd_win, up_os), n, pad)

    # Six candidate alignments (the N/2 ambiguity, times which part of
    # the 2.25 down-chirps the window found as the SFD overlaps);
    # the right one makes the header symbols, and the two full SFD
    # down-chirps, dechirp into sharp tones.
    best = None
    for half in (0, n / 2):
        tau = ((k_up - k_down) / 2 + half) % n
        cfo = ((k_up + k_down) / 2 + half + n / 2) % n - n / 2
        for m in (-1, 0, 1):
            t_sfd = sfd_win * n - tau - m * n
            a = _Aligned(x, n, cfo, t_sfd + 2.25 * n, up)
            head = a.sharpness([a.t0 + k * n for k in range(8)])
            sfd = a.sharpness([t_sfd, t_sfd + n], down=True)
            if head is None or sfd is None:
                continue
            sharp = head + sfd
            if best is None or sharp > best[0]:
                best = (sharp, a, cfo, t_sfd)
    if best is None:
        result["error"] = "packet cut short"
        return result
    _, a, cfo, t_sfd = best
    # Conjugating an inverted packet also flipped its offset's sign.
    result["cfo_hz"] = float(cfo * bw_hz / n) * (-1 if inverted else 1)
    offset = 1                  # data symbols are sent +1 (see encode)
    head = _header(a, sf, offset)
    if not head or not head["ok"]:
        result["error"] = "header checksum failed"
        return result
    result.update(length=head["length"], cr=head["cr"],
                  has_crc=head["has_crc"], fixed=head["fixed"])
    # Sync word: the two symbols before the SFD, in units of 8 bins.
    sync = []
    for k in (-2, -1):
        v = a.value_at(t_sfd + k * n)
        sync.append(None if v is None else int(round(v / 8)) % 16)
    if None not in sync:
        result["sync_word"] = (sync[0] << 4) | sync[1]

    # Payload blocks.
    nib = list(head["extra"])
    need = 2 * head["length"] + (4 if head["has_crc"] else 0)
    sf_app = sf - 2 if ldro else sf
    cw_len = 4 + head["cr"]
    k = 8
    while len(nib) < need:
        values = [a.tracked(k + i) for i in range(cw_len)]
        if None in values:
            result["error"] = "packet cut short"
            break
        got, fixed = _demod_block(values, sf, head["cr"], ldro, offset)
        nib += got
        result["fixed"] += fixed
        k += cw_len
    nbytes = min(head["length"], len(nib) // 2)
    raw = bytes(nib[2 * i] | (nib[2 * i + 1] << 4) for i in range(nbytes))
    result["payload"] = whiten(raw)
    if head["has_crc"] and len(nib) >= need:
        c = nib[2 * head["length"]:2 * head["length"] + 4]
        sent = c[0] | (c[1] << 4) | (c[2] << 8) | (c[3] << 12)
        result["crc_ok"] = sent == payload_crc(result["payload"])
        result["ok"] = result["crc_ok"]
        if not result["crc_ok"]:
            result["error"] = "payload CRC failed"
    elif not head["has_crc"] and len(nib) >= need:
        result["ok"] = True
    return result
