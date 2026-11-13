#!/usr/bin/env python3
"""test_hamscan.py — self-test for hamscan.py and bandplan.py.

No hardware needed: band-plan lookups are checked against known answers,
and the signal measurement runs on synthetic CW, SSB and FM signals.

    python3 test_hamscan.py

Exits 0 when every test passes, 1 otherwise.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import signal

import bandplan
from bandplan import classify, min_class
from cwscan import averaged_psd
from hamscan import (FFT_SIZE, SAMPLE_RATE, coverage, describe,
                     load_birdies, measure_signals, merge_signals,
                     reach_label, reachable_pieces, save_birdies, spur_kind)

SR = SAMPLE_RATE
N = SR // FFT_SIZE * FFT_SIZE  # one default 1 s step, whole FFTs
T = np.arange(N) / SR


# --------------------------------------------------------------------------
# Synthetic signal builders
# --------------------------------------------------------------------------

def _noise(rng):
    return (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / np.sqrt(2)


def _voice(rng):
    """Voice-like audio: noise band-limited to 300-3000 Hz, unit RMS."""
    sos = signal.butter(6, [300, 3000], btype="bandpass", fs=SR, output="sos")
    audio = signal.sosfilt(sos, rng.standard_normal(N))
    return audio / np.std(audio)


def make_cw(offset_hz, amp=0.05):
    return amp * np.exp(2j * np.pi * offset_hz * T)


def make_usb(rng, offset_hz, amp=0.3):
    """Upper sideband voice, 300-3000 Hz above offset_hz."""
    return signal.hilbert(_voice(rng)) * amp * np.exp(2j * np.pi * offset_hz * T)


def make_nbfm(rng, offset_hz, amp=0.3, deviation_hz=2500):
    """Narrowband FM voice centred on offset_hz."""
    phase = 2 * np.pi * np.cumsum(deviation_hz * _voice(rng)) / SR
    return amp * np.exp(1j * (2 * np.pi * offset_hz * T + phase))


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_privileges():
    """Lowest license class for known frequencies and modes."""
    cases = [
        (7.030e6, "CW", "Technician"),       # Tech CW slice
        (7.030e6, "data (RTTY/PSK/etc)", "General"),
        (7.010e6, "CW", "Extra"),
        (7.150e6, "SSB", "Extra"),
        (7.200e6, "SSB", "General"),
        (3.750e6, "SSB", "Extra"),
        (14.200e6, "SSB", "Extra"),
        (14.300e6, "SSB", "General"),
        (21.240e6, "SSB", "Extra"),
        (28.400e6, "SSB", "Technician"),
        (28.600e6, "SSB", "General"),
        (10.120e6, "CW", "General"),
        (5.332e6, "SSB", "General"),          # 60 m channel 1
        (5.340e6, "SSB", None),               # between 60 m channels
        (144.050e6, "CW", "Technician"),
        (146.520e6, "FM", "Technician"),
        (1296.100e6, "CW", "Technician"),
        (7.400e6, "SSB", None),               # broadcast, not ham
        (136.0e3, "CW", "General"),           # 2200 m
        (475.0e3, "data (RTTY/PSK/etc)", "General"),  # 630 m
        (219.5e6, "SSB/data", "Technician"),  # 219 MHz, data only
        (219.5e6, "CW", "Technician"),        # CW not allowed; data is
        (2304.1e6, "CW", "Technician"),
        (2350.0e6, "CW", None),               # gap in 13 cm
        (10368.1e6, "CW", "Technician"),
        (3400.0e6, "CW", None),               # 9 cm removed
    ]
    for freq, mode, want in cases:
        got = min_class(freq, mode)
        assert got == want, "%s %s: %s, want %s" % (freq, mode, got, want)


def test_band_class():
    """Lowest class per band, with Technician CW-only bands marked."""
    want = {"2200m": "General", "160m": "General", "80m": "Technician (CW only)",
            "40m": "Technician (CW only)", "20m": "General",
            "15m": "Technician (CW only)", "10m": "Technician",
            "2m": "Technician", "3cm": "Technician"}
    for band, cls in want.items():
        got = bandplan.band_class(band)
        assert got == cls, "%s: %s, want %s" % (band, got, cls)


def _band(name):
    return next(b for b in bandplan.BANDS if b[0] == name)


def test_dongle_coverage():
    """Bands are clipped to what each tuner chip can reach."""
    r820t = coverage(5)
    assert reach_label(_band("40m"), r820t) == "no"
    assert reach_label(_band("2m"), r820t) == "yes"
    assert reach_label(_band("13cm"), r820t) == "no"
    # Direct sampling adds HF below the tuner, with ds mode 2 (Q).
    r820t_q = coverage(5, "q")
    assert reachable_pieces(_band("40m"), r820t_q) == [(7.0e6, 7.3e6, 2)]
    assert reachable_pieces(_band("2m"), r820t_q) == [(144e6, 148e6, 0)]
    # An R828D (V4) gets HF without direct sampling.
    assert reachable_pieces(_band("20m"), coverage(6)) == [(14.0e6, 14.35e6, 0)]
    # E4000 has a gap at 1100-1250 MHz, so 23 cm is only partly reachable;
    # it starts at 52 MHz, so it gets the top of 6 m and none of 10 m.
    e4000 = coverage(1)
    assert reachable_pieces(_band("23cm"), e4000) == [(1250e6, 1300e6, 0)]
    assert reach_label(_band("23cm"), e4000) == "partly"
    assert reachable_pieces(_band("6m"), e4000) == [(52e6, 54e6, 0)]
    assert reach_label(_band("10m"), e4000) == "no"
    # FC2580 starts at 146 MHz: half of 2 m.
    assert reachable_pieces(_band("2m"), coverage(4)) == [(146e6, 148e6, 0)]
    # Unknown tuner: claim nothing.
    assert coverage(99) == [] and reach_label(_band("2m"), []) == "no"


def test_classify():
    """Mode guesses from band plan and width."""
    cases = [
        (14.0755e6, 60, "FT8"),
        (7.0755e6, 60, "FT8"),
        (14.0971e6, 60, "WSPR"),
        (7.0401e6, 60, "WSPR"),
        (7.030e6, 190, "CW"),
        (7.0401e6, 3000, "data (RTTY/PSK/etc)"),  # too wide for WSPR
        (14.250e6, 4100, "SSB"),
        (3.885e6, 7000, "AM/FM"),
        (50.050e6, 190, "CW"),
        (144.200e6, 4100, "SSB"),
        (146.520e6, 13000, "FM"),
        (146.520e6, 190, "FM carrier (no audio)"),
        (435.500e6, 190, "satellite beacon/CW"),
        (426.250e6, 2e6, "wideband (ATV/data)"),
    ]
    for freq, width, want in cases:
        got = classify(freq, width)
        assert got == want, "%s, %s Hz: %r, want %r" % (freq, width, got, want)


def test_tables_consistent():
    """Segments are ordered, don't overlap, and lie inside a band."""
    for name, table in (("PRIVILEGES", bandplan.PRIVILEGES),
                        ("USAGE", bandplan.USAGE)):
        for (lo1, hi1, *_), (lo2, hi2, *_) in zip(table, table[1:]):
            assert lo1 < hi1 <= lo2 < hi2, "%s out of order near %s" % (name, lo1)
        for lo, hi, *_ in table:
            band = bandplan.band_for(lo)
            assert band and bandplan.band_for(hi) == band, \
                "%s segment %s-%s outside a band" % (name, lo, hi)


def test_measure_widths():
    """CW, SSB and FM in one spectrum come out with the right widths."""
    for seed in range(3):
        rng = np.random.default_rng(seed)
        iq = (_noise(rng) + make_cw(100e3) + make_usb(rng, 300e3)
              + make_nbfm(rng, 600e3))
        found = measure_signals(*averaged_psd(iq, SR, FFT_SIZE), 5.0)
        assert len(found) == 3, "seed %d: found %s" % (seed, found)
        by_freq = sorted(found, key=lambda s: s["freq_hz"])
        cw, ssb, fm = (s["width_hz"] for s in by_freq)
        assert cw < bandplan.NARROW_HZ, "CW width %s" % cw
        assert bandplan.NARROW_HZ <= ssb < bandplan.VOICE_HZ, "SSB width %s" % ssb
        assert bandplan.VOICE_HZ <= fm < bandplan.NBFM_HZ, "FM width %s" % fm
        assert abs(by_freq[0]["freq_hz"] - 100e3) < 100
        assert abs(by_freq[2]["freq_hz"] - 600e3) < 2000


def test_quiet_on_noise():
    """Pure noise stays quiet at the default threshold."""
    for seed in range(5):
        rng = np.random.default_rng(100 + seed)
        found = measure_signals(*averaged_psd(_noise(rng), SR, FFT_SIZE), 5.0)
        assert found == [], "false positives (seed %d): %s" % (seed, found)


def test_merge_signals():
    """An FM signal seen by two steps merges; a nearby CW signal doesn't
    merge with another CW signal 1 kHz away."""
    hits = [
        {"freq_hz": 146.5200e6, "snr_db": 20.0, "width_hz": 13000},
        {"freq_hz": 146.5230e6, "snr_db": 15.0, "width_hz": 9000},
        {"freq_hz": 7.0300e6, "snr_db": 10.0, "width_hz": 190},
        {"freq_hz": 7.0310e6, "snr_db": 12.0, "width_hz": 190},
    ]
    merged = merge_signals(hits)
    assert [h["freq_hz"] for h in merged] == [7.0300e6, 7.0310e6, 146.5200e6], \
        "merged to %s" % merged


def test_spurs():
    """14.4 MHz harmonics and saved birdies are flagged; others aren't."""
    for freq in (144.0e6, 432.0e6, 446.4e6, 907.2e6, 1296.002e6):
        assert spur_kind(freq) == "crystal", freq
    for freq in (144.010e6, 146.52e6, 440.0e6):
        assert spur_kind(freq) is None, freq
    assert spur_kind(440.064e6, birdies=[440.0641e6]) == "birdie"
    hit = describe({"freq_hz": 144.0e6, "snr_db": 20.0, "width_hz": 120})
    assert hit["spur"] == "crystal" and hit["mode"] == "spur (crystal)", hit
    assert hit["band"] == "2m" and hit["license"] == "Technician", hit
    hit = describe({"freq_hz": 146.52e6, "snr_db": 20.0, "width_hz": 13000})
    assert hit["spur"] is None and hit["mode"] == "FM", hit


def test_birdies_file():
    """Birdies saved by --calibrate load back, comments and all."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "birdies.txt")
        assert load_birdies(path) == [], "missing file should be empty"
        save_birdies(path, [{"freq_hz": 440.064e6}, {"freq_hz": 909.184e6}])
        assert load_birdies(path) == [440.064e6, 909.184e6]


TESTS = [
    ("license privileges", test_privileges),
    ("lowest class per band", test_band_class),
    ("dongle coverage per tuner", test_dongle_coverage),
    ("mode guesses", test_classify),
    ("band-plan tables consistent", test_tables_consistent),
    ("CW/SSB/FM widths measured", test_measure_widths),
    ("quiet on pure noise", test_quiet_on_noise),
    ("hits merge across steps", test_merge_signals),
    ("dongle spurs flagged", test_spurs),
    ("birdies file round-trip", test_birdies_file),
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
