#!/usr/bin/env python3
"""test_lora.py — self-test for lora.py, loraregions.py and lorascan.py.

No hardware needed: synthetic LoRa packets (every bandwidth, SF7-12,
normal and inverted I/Q, weak and strong) go through the detector, and a
fake dongle drives the sweep and the gapless stream.

    python3 test_lora.py

Exits 0 when every test passes, 1 otherwise.
"""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import lora
import lorademod
import lorascan
import meshtastic
from loraregions import US915

FS = lora.SAMPLE_RATE


# --------------------------------------------------------------------------
# Synthetic signals
# --------------------------------------------------------------------------

def packet(sf, bw, offset_hz=0.0, n_data=30, preamble=8, invert=False,
           seed=0):
    """A LoRa packet at FS: preamble up-chirps, two sync symbols, two
    down-chirps, then random data symbols. invert flips I/Q."""
    rng = np.random.default_rng(seed)
    n = 2 ** sf
    spc = FS / bw  # samples per chip
    syms = ([0] * preamble + [8, 16] + ["down"] * 2
            + list(rng.integers(0, n, n_data)))
    freq = []
    for s in syms:
        k = np.arange(int(round(n * spc))) / spc
        if s == "down":
            freq.append(bw * (0.5 - k / n))
        else:
            freq.append(bw * (((k + s) % n) / n - 0.5))
    freq = np.concatenate(freq)
    if invert:
        freq = -freq
    return np.exp(1j * np.cumsum(2 * np.pi * (freq + offset_hz) / FS))


def noise(seconds, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    return ((rng.standard_normal(n) + 1j * rng.standard_normal(n))
            / np.sqrt(2)).astype(np.complex64)


def add(iq, sig, at_s, snr_db, bw):
    """Add sig at time at_s with this SNR in its bandwidth (unit noise)."""
    amp = np.sqrt(10 ** (snr_db / 10) * bw / FS)
    a = int(at_s * FS)
    n = min(len(sig), len(iq) - a)
    iq[a:a + n] += (amp * sig[:n]).astype(np.complex64)
    return iq


def airtime(sf, bw, n_data=30, preamble=8):
    return (preamble + 4 + n_data) * 2 ** sf / bw


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_chirp_reference():
    """A dechirped clean up-chirp is one FFT bin; the generator's data
    symbols land on their own bin."""
    for sf in (7, 10):
        for sym in (0, 5, 2 ** sf - 1):
            d = np.fft.fft(lora.upchirp(sf, symbol=sym) * np.conj(lora.upchirp(sf)))
            assert np.argmax(np.abs(d)) == sym, (sf, sym)
            assert np.abs(d).max() ** 2 > 0.99 * np.sum(np.abs(d) ** 2)


def test_every_sf_and_bw():
    """SF7-12 at each bandwidth, at 0 dB: right SF, BW, direction, time
    and length."""
    cases = [(sf, 125_000) for sf in range(7, 13)] + [
        (7, 62_500), (10, 62_500), (7, 250_000), (10, 250_000),
        (8, 500_000), (12, 500_000)]
    for sf, bw in cases:
        iq = add(noise(2.0, seed=sf), packet(sf, bw, 250e3, seed=sf),
                 0.3, 0.0, bw)
        found = lora.analyse(iq)
        assert len(found) == 1, (sf, bw, found)
        f = found[0]
        assert (f["sf"], f["bw_hz"], f["chirp"]) == (sf, bw, "up"), (sf, bw, f)
        assert abs(f["freq_hz"] - 250e3) < 8e3, f
        assert abs(f["start_s"] - 0.3) < 0.002, f
        expect = min(airtime(sf, bw), 2.0 - 0.3)
        assert abs(f["duration_s"] - expect) < 0.003, (expect, f)


def test_inverted_iq():
    """Inverted I/Q (LoRaWAN downlink style) reads as down-chirps."""
    iq = add(noise(1.0), packet(8, 500_000, -300e3, invert=True), 0.2, 3, 500e3)
    (f,) = lora.analyse(iq)
    assert (f["sf"], f["chirp"], f["bw_hz"]) == (8, "down", 500_000), f


def test_weak_packets():
    """Packets at -4 dB SNR are still found whole and identified (SF7
    needs about 0 dB, see test_every_sf_and_bw)."""
    for sf, bw in ((9, 125_000), (11, 250_000)):
        iq = add(noise(2.0, seed=sf), packet(sf, bw, 100e3, seed=sf),
                 0.4, -4.0, bw)
        found = lora.analyse(iq)
        assert len(found) == 1, found
        f = found[0]
        assert (f["sf"], f["bw_hz"]) == (sf, bw), f
        assert abs(f["duration_s"] - airtime(sf, bw)) < 0.03, f
        assert -6 < f["snr_db"] < -2, f


def test_quiet_on_noise():
    """Two seconds of noise gives nothing (several seeds)."""
    for seed in range(4):
        found = lora.analyse(noise(2.0, seed=100 + seed))
        assert found == [], found


def test_not_lora():
    """A wideband FSK burst is found but not called LoRa; a narrow
    carrier isn't even a burst."""
    rng = np.random.default_rng(5)
    bits = np.repeat(rng.integers(0, 2, 2000) * 2 - 1, 120)  # 20 kb/s
    fsk = np.exp(1j * np.cumsum(2 * np.pi * 50e3 * bits / FS))
    iq = add(noise(1.0), fsk, 0.2, 10, 150e3)
    found = lora.analyse(iq)
    assert len(found) == 1 and not found[0]["is_lora"], found
    t = np.arange(int(0.5 * FS)) / FS
    iq = add(noise(1.0), np.exp(2j * np.pi * 200e3 * t), 0.2, 30, 125e3)
    assert lora.analyse(iq) == [], "carrier detected as a burst"


def test_too_short():
    """Five clean chirps are less than a preamble: not a LoRa packet."""
    iq = add(noise(1.0), packet(9, 125_000, 0, n_data=0, preamble=1)[
        :int(5 * 2 ** 9 / 125e3 * FS)], 0.2, 10, 125e3)
    found = lora.analyse(iq)
    assert len(found) == 1 and not found[0]["is_lora"], found


def test_side_by_side():
    """Two packets at once on neighbouring LoRaWAN channels (200 kHz
    apart) come out as two."""
    iq = noise(1.0)
    add(iq, packet(7, 125_000, -100e3, seed=1), 0.2, 6, 125e3)
    add(iq, packet(9, 125_000, 100e3, seed=2), 0.25, 6, 125e3)
    found = sorted(lora.analyse(iq), key=lambda f: f["freq_hz"])
    assert [f["sf"] for f in found] == [7, 9], found
    for f, expect in zip(found, (-100e3, 100e3)):
        assert abs(f["freq_hz"] - expect) < 5e3, f


def test_region_channels():
    """US915 channel names, LoRaWAN first, BW-gated."""
    assert US915.channels(902.3e6, 125_000) == \
        ["LoRaWAN uplink 0 (sub-band 1)", "Meshtastic slot 3"]
    assert US915.channels(904.31e6, 125_000)[0] == \
        "LoRaWAN uplink 10 (sub-band 2)"
    assert US915.channels(904.6e6, 500_000)[0] == \
        "LoRaWAN uplink 65 (sub-band 2)"
    assert US915.channels(923.3e6, 500_000) == ["LoRaWAN downlink 0"]
    assert US915.channels(906.875e6, 250_000) == \
        ["Meshtastic slot 20 (LongFast default)"]
    assert US915.channels(910.525e6, 62_500) == ["MeshCore 0 (US default)"]
    assert US915.channels(904.4e6, 125_000) == [], "between channels"
    assert US915.channels(904.3e6, 250_000) == [], "wrong bandwidth"


def test_mesh_flags():
    """--meshtastic / --meshcore centre on the US defaults; they can't be
    combined with each other or with --center."""
    parser = lorascan.build_parser()
    for flag, freq, bw, sf in (("--meshtastic", 906.875e6, 250_000, 11),
                               ("--meshcore", 910.525e6, 62_500, 7)):
        args = parser.parse_args([flag])
        lorascan.apply_preset(args, parser)
        assert float(args.center) == freq, args.center
        assert (args.preset.bw_hz, args.preset.sf) == (bw, sf), args.preset
    args = parser.parse_args([])
    lorascan.apply_preset(args, parser)
    assert args.center is None and args.preset is None
    for bad in (["--meshcore", "--center", "904M"],
                ["--meshcore", "--meshtastic"]):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                args = parser.parse_args(bad)
                lorascan.apply_preset(args, parser)
        except SystemExit:
            pass
        else:
            raise AssertionError("accepted %s" % bad)


def test_dwell():
    """400 ms limit below 500 kHz; none at 500 kHz."""
    assert US915.dwell_check(0.370, 125_000) == "ok"
    assert US915.dwell_check(1.32, 125_000) == "over limit"
    assert US915.dwell_check(0.2, 125_000, cut_off=True) == "ok?"
    assert US915.dwell_check(1.5, 500_000) == "n/a"


def test_sweep_tiling():
    """Sweep shares tile the band exactly and every share plus a 500 kHz
    signal fits inside the trusted span."""
    centres, half = lorascan.sweep_centres(US915.low_hz, US915.high_hz)
    assert abs(centres[0] - half - US915.low_hz) < 1
    assert abs(centres[-1] + half - US915.high_hz) < 1
    assert all(abs(b - a - 2 * half) < 1 for a, b in zip(centres, centres[1:]))
    assert half + lora.LORA_BANDWIDTHS[-1] / 2 <= lorascan.USABLE_HZ
    assert len(centres) == 19, len(centres)


class FakeSdr:
    """Stands in for RtlSdr: each async read plays `scene(centre)` in
    blocks, then noise, until cancelled."""

    def __init__(self, scene):
        self.scene = scene
        self.center_freq = 0.0
        self.cancelled = False

    def read_samples_async(self, callback, n):
        self.cancelled = False
        iq = self.scene(self.center_freq)
        i = 0
        while not self.cancelled:
            block = iq[i:i + n]
            if len(block) < n:
                block = noise(n / FS, seed=i)[:n]
            callback(block, None)
            i += n

    def cancel_read_async(self):
        self.cancelled = True


class Collect:
    def __init__(self):
        self.facts = []

    def report(self, facts, iq):
        self.facts.append(facts)


def test_stream_gapless():
    """Stream.read hands back consecutive samples with nothing lost or
    repeated, across odd read sizes."""
    ramp = np.arange(lorascan.READ_BLOCK * 5).astype(np.complex64)
    with lorascan.Stream(FakeSdr(lambda c: ramp)) as stream:
        _, a = stream.read(100_000)
        _, b = stream.read(300_000)
        _, c = stream.read(500_000)
    got = np.concatenate([a, b, c]).real
    assert np.array_equal(got, np.arange(900_000)), "samples lost or repeated"


def test_sweep_end_to_end():
    """A sweep over a fake dongle reports a packet exactly once, with
    its absolute frequency and channel, even where steps overlap."""
    target = 904.3e6

    def scene(centre):
        iq = noise(1.2, seed=int(centre) % 1000)
        if abs(target - centre) < lorascan.USABLE_HZ:
            add(iq, packet(8, 125_000, target - centre), 0.4, 6, 125e3)
        return iq

    args = argparse.Namespace(start="903M", stop="906M", dwell=1.0, passes=1,
                              threshold=lora.DEFAULT_THRESHOLD_DB)
    out = Collect()
    lorascan.run_sweep(FakeSdr(scene), args, US915, out)
    assert len(out.facts) == 1, out.facts
    f = out.facts[0]
    assert abs(f["freq_hz"] - target) < 5e3 and f["sf"] == 8, f
    assert f["channels"][0] == "LoRaWAN uplink 10 (sub-band 2)", f
    assert f["dwell"] == "ok", f


def test_watch_carry_over():
    """In watch mode a packet crossing a segment boundary is reported
    once and whole, and a short packet finished meanwhile on another
    channel isn't reported twice."""
    seg = int(lorascan.SEGMENT_S * FS)
    iq = noise(2 * lorascan.SEGMENT_S + 1.0, seed=9)
    long_sf = 11
    add(iq, packet(long_sf, 125_000, -200e3, seed=3), 1.6, 6, 125e3)
    add(iq, packet(7, 125_000, 300e3, seed=4), 1.8, 6, 125e3)
    args = argparse.Namespace(threshold=lora.DEFAULT_THRESHOLD_DB)
    out = Collect()
    carry, t_carry, recent = np.zeros(0, np.complex64), None, []
    t = 0.0
    for i in range(0, len(iq) - seg + 1, seg):
        fresh = iq[i:i + seg]
        t0 = t_carry if len(carry) else t
        carry, t_carry, recent = lorascan.watch_segment(
            np.concatenate([carry, fresh]), t0, 915e6, args, US915, out,
            recent)
        t += seg / FS
    sfs = sorted(f["sf"] for f in out.facts)
    assert sfs == [7, long_sf], out.facts
    long = next(f for f in out.facts if f["sf"] == long_sf)
    assert abs(long["time"] - 1.6) < 0.005, long
    assert abs(long["duration_s"] - airtime(long_sf, 125e3)) < 0.005, long
    assert not long["cut_off"] and long["dwell"] == "over limit", long


def test_save_iq():
    """--save-iq writes the packet at 0 Hz plus facts that load back."""
    iq = add(noise(1.0), packet(7, 125_000, 300e3), 0.2, 10, 125e3)
    (burst,) = lora.analyse(iq)
    facts = lorascan.describe(burst, 915e6, 1_700_000_000.0, US915)
    with tempfile.TemporaryDirectory() as tmp:
        path = lorascan.save_iq(tmp, iq, facts, FS)
        piece = np.load(path)
        with open(path[:-4] + ".json") as fh:
            meta = json.load(fh)
    assert meta["sf"] == 7 and meta["sample_rate"] == FS, meta
    chips = lora.to_baseband(piece, FS, 0.0, 125_000)
    assert lora.dechirp_score(chips[int(0.005 * 125e3):], 7) > 15, \
        "saved samples aren't centred on the packet"


def test_fact_row():
    """The printed row carries the facts."""
    iq = add(noise(1.0), packet(9, 125_000, 300e3), 0.2, 10, 125e3)
    (burst,) = lora.analyse(iq)
    row = lorascan.fmt_row(lorascan.describe(burst, 904.0e6, 0.0, US915))
    for text in ("904.3000 MHz", "125 kHz", "SF9", "up", "ok",
                 "LoRaWAN uplink 10"):
        assert text in row, (text, row)


def test_whitening_and_crc():
    """LoRa whitening starts FF FE FC F8 F0 E1 C2 85 0B 17 2F 5E; CRC-16
    is CCITT with init 0 (check value of '123456789' is 0x31C3)."""
    assert lorademod.WHITENING[:12].hex() == "fffefcf8f0e1c2850b172f5e"
    assert lorademod.crc16(b"123456789") == 0x31C3


def test_hamming_and_interleave():
    """Every nibble survives Hamming at each CR, CR 4/7 and 4/8 fix any
    one flipped bit, and interleaving round-trips."""
    for cr in range(1, 5):
        for nib in range(16):
            assert lorademod.hamming_decode(
                lorademod.hamming_encode(nib, cr), cr)[0] == nib
    for cr in (3, 4):
        for nib in range(16):
            cw = lorademod.hamming_encode(nib, cr)
            for bit in range(4 + cr):
                assert lorademod.hamming_decode(cw ^ (1 << bit), cr)[0] == nib
    rng = np.random.default_rng(0)
    for sf_app, cw_len in ((7, 5), (10, 8), (5, 8)):
        cws = list(rng.integers(0, 2 ** cw_len, sf_app))
        values = lorademod.interleave(cws, sf_app, cw_len, False)
        assert lorademod.deinterleave(values, sf_app, cw_len) == cws


def lora_packet(payload, sf, bw, cr=1, sync=0x12, invert=False,
                offset_hz=250e3, snr_db=10, preamble=8, seed=0):
    """A whole encoded packet in noise, starting 5 ms in."""
    syms = lorademod.encode(payload, sf, cr, True, lorademod.use_ldro(sf, bw))
    sig = lorademod.modulate(syms, sf, bw, FS, sync_word=sync,
                             preamble=preamble, offset_hz=offset_hz,
                             invert=invert)
    iq = noise(len(sig) / FS + 0.05, seed=seed)
    return add(iq, sig, 0.005, snr_db, bw)


def test_decode_round_trip():
    """Payloads come back byte for byte across SF, bandwidth, coding
    rate, I/Q inversion, LDRO and frequency offset, at 0 dB SNR."""
    rng = np.random.default_rng(1)
    cases = [(7, 125_000, 1, False), (9, 125_000, 4, False),
             (8, 500_000, 3, True), (11, 250_000, 2, False),
             (11, 125_000, 1, False),  # LDRO
             (7, 62_500, 1, False)]
    for sf, bw, cr, inv in cases:
        payload = bytes(rng.integers(0, 256, 30).tolist())
        iq = lora_packet(payload, sf, bw, cr, sync=0x34, invert=inv,
                         offset_hz=250e3 + 3e3, snr_db=0, seed=sf)
        r = lorademod.decode(iq, FS, 250e3, bw, sf, inverted=inv)
        assert r["ok"] and r["payload"] == payload, (sf, bw, cr, inv, r)
        assert r["sync_word"] == 0x34 and r["cr"] == cr, r
        assert abs(r["cfo_hz"] - 3e3) < bw / 2 ** sf, r


def test_meshtastic_over_lora():
    """A LongFast text message, sent as LoRa (sync word 0x2B), is found,
    demodulated, decrypted and printed by lorascan."""
    pkt = meshtastic.encode(meshtastic.BROADCAST, 0xDEADBEEF, 0x01020304, 1,
                            b"hi from the test")
    iq = lora_packet(pkt, 11, 250_000, cr=1, sync=0x2B, preamble=16,
                     snr_db=6)
    (burst,) = [b for b in lora.analyse(iq) if b["is_lora"]]
    facts = lorascan.describe(burst, 906.625e6, 0.0, US915)
    lorascan.decode_packet(facts, iq)
    lines = "\n".join(lorascan.fmt_decoded(facts))
    for text in ("sync 0x2B (Meshtastic)", "CRC ok", "!deadbeef -> ^all",
                 "[LongFast]", 'TEXT_MESSAGE: "hi from the test"'):
        assert text in lines, (text, lines)
    assert facts["channels"][0].startswith("Meshtastic slot 20"), facts


def test_decode_saved_file():
    """A packet saved with --save-iq decodes again with --decode-file."""
    payload = b"saved for later"
    iq = lora_packet(payload, 8, 125_000, cr=2, seed=4)
    (burst,) = [b for b in lora.analyse(iq) if b["is_lora"]]
    facts = lorascan.describe(burst, 915e6, 1_700_000_000.0, US915)
    with tempfile.TemporaryDirectory() as tmp:
        path = lorascan.save_iq(tmp, iq, facts, FS)
        args = lorascan.build_parser().parse_args(["--decode-file", path])
        args.channels = []
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = lorascan.cmd_decode_files(args)
    assert status == 0, out.getvalue()
    assert "CRC ok" in out.getvalue() and payload.hex()[:20] in out.getvalue()


TESTS = [
    ("chirp generator", test_chirp_reference),
    ("SF7-12 at every bandwidth", test_every_sf_and_bw),
    ("inverted I/Q seen as down-chirps", test_inverted_iq),
    ("weak packets (-4 dB)", test_weak_packets),
    ("quiet on pure noise", test_quiet_on_noise),
    ("FSK and carriers aren't LoRa", test_not_lora),
    ("5-symbol chirp burst isn't a packet", test_too_short),
    ("neighbouring channels kept apart", test_side_by_side),
    ("US915 channel names", test_region_channels),
    ("--meshtastic / --meshcore", test_mesh_flags),
    ("US915 dwell limit", test_dwell),
    ("sweep steps tile the band", test_sweep_tiling),
    ("gapless stream", test_stream_gapless),
    ("sweep end to end (fake dongle)", test_sweep_end_to_end),
    ("watch carries packets across segments", test_watch_carry_over),
    ("--save-iq round trip", test_save_iq),
    ("fact row", test_fact_row),
    ("whitening and CRC", test_whitening_and_crc),
    ("Hamming and interleaving", test_hamming_and_interleave),
    ("decode round trip at 0 dB", test_decode_round_trip),
    ("Meshtastic message over LoRa", test_meshtastic_over_lora),
    ("--decode-file", test_decode_saved_file),
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
