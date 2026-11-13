#!/usr/bin/env python3
"""test_synth.py — self-test for cwscan.py and morse.py.

No hardware needed: every test builds synthetic IQ (or a synthetic keying
envelope) in memory and runs the same detection / demodulation / decoding
code the real modes use.

    python3 test_synth.py

Exits 0 when every test passes, 1 otherwise.
"""

import os
import sys
import tempfile
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import morse
from cwscan import (DEFAULT_FFT_SIZE, DEFAULT_THRESHOLD_DB, demodulate_cw,
                    detect_signals, envelope_to_binary, merge_hits,
                    scan_steps, usable_band, write_wav)

SR = 250_000  # same sample rate the scanner uses by default
DWELL = 253952  # samples in one default 1 s scan step (whole FFTs)


# --------------------------------------------------------------------------
# Synthetic signal builders
# --------------------------------------------------------------------------

def _add_noise(iq, snr_db):
    """Add complex Gaussian noise for a per-sample SNR of snr_db dB.

    The signal is assumed to have unit power while the key is down, so the
    noise standard deviation per quadrature is 10**(-snr/20) / sqrt(2).
    """
    sigma = 10 ** (-snr_db / 20.0)
    noise = (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq)))
    return iq + noise * (sigma / np.sqrt(2))


def make_cw_iq(text, wpm, offset_hz, sample_rate=SR, snr_db=15.0):
    """Synthetic IQ of `text` keyed in Morse at `offset_hz` from centre."""
    env = morse.keying_envelope(text, wpm, sample_rate)
    t = np.arange(len(env)) / sample_rate
    iq = env * np.exp(2j * np.pi * offset_hz * t)
    return _add_noise(iq, snr_db)


def make_carrier_iq(offset_hz, duration_s, sample_rate=SR, snr_db=12.0):
    """Synthetic IQ of a steady unmodulated carrier at `offset_hz`."""
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    iq = np.exp(2j * np.pi * offset_hz * t)
    return _add_noise(iq, snr_db)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_detector_finds_keyed_and_steady_signals():
    """Two signals in one chunk: a steady carrier and a keyed 'CQ'."""
    np.random.seed(42)
    dur = 65536 / SR  # one scanner read, ~0.26 s
    carrier = make_carrier_iq(-30_000, dur, snr_db=12.0)
    cw = make_cw_iq("CQ", 20, 45_000, snr_db=15.0)
    # Match lengths and add the two signals together.
    n = min(len(carrier), len(cw))
    iq = carrier[:n] + cw[:n]
    found = detect_signals(iq, SR, DEFAULT_FFT_SIZE, threshold_db=10.0)
    freqs = sorted(c["freq_hz"] for c in found)
    assert len(freqs) == 2, "expected 2 signals, found %d: %s" % (len(freqs), freqs)
    assert abs(freqs[0] - (-30_000)) < 150, "carrier off: %s" % freqs
    assert abs(freqs[1] - 45_000) < 150, "CW signal off: %s" % freqs
    assert all(c["snr_db"] > 10 for c in found), "weak detection: %s" % found


def test_detector_quiet_band_is_quiet():
    """Pure noise should produce no candidates at the default threshold."""
    np.random.seed(7)
    iq = _add_noise(np.zeros(65536, dtype=complex), snr_db=0.0)
    found = detect_signals(iq, SR, DEFAULT_FFT_SIZE, threshold_db=10.0)
    assert found == [], "false positives on noise: %s" % found


def test_detector_finds_weak_signal():
    """A carrier ~6 dB above the floor is found at the default threshold
    and not mistaken for anything else over a full dwell of noise."""
    np.random.seed(5)
    iq = make_carrier_iq(41_000, DWELL / SR, snr_db=-28.0)
    found = detect_signals(iq, SR, DEFAULT_FFT_SIZE, DEFAULT_THRESHOLD_DB)
    assert len(found) == 1, "expected 1 signal, found %s" % found
    assert abs(found[0]["freq_hz"] - 41_000) < 150, "off: %s" % found


def test_detector_quiet_at_default_threshold():
    """Full dwells of pure noise stay quiet at the low default threshold."""
    for seed in range(10):
        np.random.seed(seed)
        iq = _add_noise(np.zeros(DWELL, dtype=complex), snr_db=0.0)
        found = detect_signals(iq, SR, DEFAULT_FFT_SIZE, DEFAULT_THRESHOLD_DB)
        assert found == [], "false positives (seed %d): %s" % (seed, found)


def test_scan_ignores_dc_spike():
    """A DC spike is found by the detector but is outside the usable band,
    while a real signal inside it is kept."""
    np.random.seed(11)
    t = np.arange(DWELL) / SR
    iq = _add_noise(np.full(DWELL, 0.3, dtype=complex)  # DC spike
                    + 0.05 * np.exp(2j * np.pi * 60_000 * t), snr_db=0.0)
    found = detect_signals(iq, SR, DEFAULT_FFT_SIZE, DEFAULT_THRESHOLD_DB)
    assert any(abs(c["freq_hz"]) < 100 for c in found), \
        "test setup: DC spike not visible: %s" % found
    low, high = usable_band(SR)
    kept = [c["freq_hz"] for c in found if low <= c["freq_hz"] < high]
    assert len(kept) == 1 and abs(kept[0] - 60_000) < 150, \
        "kept %s" % kept


def test_scan_steps_cover_without_gaps():
    """Steps tile the range edge to edge, never using a step's centre."""
    low, high = usable_band(SR)
    for start, stop in ((7.0e6, 7.04e6), (14.0e6, 14.07e6), (540e3, 1.6e6)):
        centres = scan_steps(start, stop, SR)
        assert abs(centres[0] + low - start) < 1e-6, "first step %s" % centres[0]
        assert centres[-1] + high >= stop, "range end not covered"
        for a, b in zip(centres, centres[1:]):
            assert abs((a + high) - (b + low)) < 1e-6, \
                "gap/overlap between %s and %s" % (a, b)


def test_merge_hits():
    """Hits within MERGE_HZ collapse to the strongest; others stay apart."""
    hits = [(7_030_000, 6.0), (7_030_100, 9.0), (7_031_000, 5.5)]
    assert merge_hits(hits) == [(7_030_100, 9.0), (7_031_000, 5.5)]


def test_decoder_known_text():
    """Decode synthetic keying envelopes at audio rate."""
    for text, wpm in (("CQ", 20), ("73", 25), ("CQ DE TEST", 18)):
        env = morse.keying_envelope(text, wpm, sample_rate=8000)
        decoded = morse.decode_envelope(env, 8000)
        assert decoded == text, "%r at %d WPM decoded as %r" % (text, wpm, decoded)
    # Same text with the speed given explicitly.
    env = morse.keying_envelope("SOS", 30, sample_rate=8000)
    assert morse.decode_envelope(env, 8000, wpm=30) == "SOS"


def test_decoder_roundtrip_with_words():
    env = morse.keying_envelope("HELLO WORLD 73", 20, sample_rate=8000)
    assert morse.decode_envelope(env, 8000) == "HELLO WORLD 73"


def test_decoder_transitions_directly():
    """Pin the dit/dah and gap thresholds with hand-built timings."""
    unit = 0.06  # 20 WPM
    # "SOS": ... --- ...  (marks and gaps in seconds)
    transitions = []
    for symbol in "...":
        transitions += [(True, unit), (False, unit)]
    transitions += [(False, 2 * unit)]  # char gap (1 already there)
    for symbol in "---":
        transitions += [(True, 3 * unit), (False, unit)]
    transitions += [(False, 2 * unit)]
    for symbol in "...":
        transitions += [(True, unit), (False, unit)]
    assert morse.decode_transitions(transitions) == "SOS"


def test_listen_pipeline_decodes():
    """End to end on synthetic IQ: demodulate -> envelope -> decode."""
    np.random.seed(123)
    iq = make_cw_iq("CQ", 20, 0, snr_db=15.0)
    audio, env, audio_rate = demodulate_cw(iq, SR)
    assert audio_rate == 8000
    assert len(audio) > 0 and np.max(np.abs(audio)) > 0
    binary = envelope_to_binary(env, audio_rate)
    decoded = morse.decode_envelope(binary, audio_rate)
    assert decoded == "CQ", "listen pipeline decoded %r" % decoded


def test_wav_roundtrip():
    """write_wav produces a readable WAV with the expected length."""
    np.random.seed(99)
    iq = make_cw_iq("E", 20, 0, snr_db=20.0)
    audio, _, audio_rate = demodulate_cw(iq, SR)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        write_wav(path, audio, audio_rate)
        with wave.open(path, "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getframerate() == audio_rate
            assert wav.getnframes() == len(audio), \
                "frames %d != samples %d" % (wav.getnframes(), len(audio))
    finally:
        os.unlink(path)


TESTS = [
    ("detector finds keyed + steady signals", test_detector_finds_keyed_and_steady_signals),
    ("detector quiet on pure noise", test_detector_quiet_band_is_quiet),
    ("detector finds a weak (~6 dB) signal", test_detector_finds_weak_signal),
    ("detector quiet at default threshold", test_detector_quiet_at_default_threshold),
    ("scan ignores the DC spike", test_scan_ignores_dc_spike),
    ("scan steps tile the range", test_scan_steps_cover_without_gaps),
    ("nearby hits merge", test_merge_hits),
    ("decoder known text at several speeds", test_decoder_known_text),
    ("decoder round-trip with words", test_decoder_roundtrip_with_words),
    ("decoder hand-built SOS timings", test_decoder_transitions_directly),
    ("listen pipeline decodes CQ", test_listen_pipeline_decodes),
    ("WAV write/read round-trip", test_wav_roundtrip),
]


def main():
    failures = 0
    for name, fn in TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report any failure plainly
            failures += 1
            print("FAIL  %s\n      %s: %s" % (name, type(exc).__name__, exc))
        else:
            print("PASS  %s" % name)
    print()
    if failures:
        print("%d of %d tests failed." % (failures, len(TESTS)))
        return 1
    print("All %d tests passed." % len(TESTS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
