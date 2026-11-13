#!/usr/bin/env python3
"""hamscan.py — sweep the US amateur bands with an RTL-SDR and list signals.

    python3 hamscan.py                     # every band the dongle can reach
    python3 hamscan.py --bands 2m,70cm     # just these
    python3 hamscan.py --list-bands

For each signal found it prints the band, frequency, strength, width, a
likely mode, and the lowest US license class (Technician, General or
Extra) allowed to transmit that mode there.

How it works
------------
It uses the same detector as cwscan.py (averaged FFT, rolling-median noise
floor, DC spike skipped), but at 2.048 MS/s so each step covers ~920 kHz
and the wide UHF bands go quickly. The FFT is 32768 points, so bins are
still ~62 Hz -- narrow enough for CW. The noise-floor median spans 40 kHz
so that FM signals (~16 kHz wide) stand out instead of raising the floor.

Each signal's width is measured from the spectrum: the run of bins within
WIDTH_DROP_DB of its peak and above the floor, bridging small gaps (SSB
and FM spectra are lumpy). The mode guess combines that width with the
ARRL band plan and the FT8/FT4/WSPR watering holes, see bandplan.py.
It is a guess: an idle FM carrier looks like CW, and so on.

Hardware notes
--------------
The band list covers every US amateur band, 2200 m to 1 mm; what gets
scanned depends on the dongle. Its tuner chip is read at startup and each
band is clipped to what that chip can tune (see TUNERS), and bands it
can't reach at all are listed at the end. HF below the tuner is added
with --direct-sampling (RTL-SDR Blog V3) or automatically for an R828D
tuner (RTL-SDR Blog V4, which has an HF upconverter). Direct sampling is
only switched on for those frequencies, and off again above them.
--list-bands shows every band and whether the plugged-in dongle can
receive it.

Spurs ("birdies")
-----------------
A dongle hears its own clocks. Harmonics of 14.4 MHz (half its 28.8 MHz
crystal) land right on 144.000, 432.000 and 1296.000 MHz, and are always
treated as spurs. Other birdies depend on the dongle and the computer
it's plugged into, so they are learned: disconnect the antenna and run

    python3 hamscan.py --calibrate

which saves every hit to birdies.txt. Later scans hide hits within
SPUR_GUARD_HZ of a crystal harmonic or a saved birdie, and say how many
they hid; --show-spurs lists them, marked.
"""

import argparse
import os
import sys
import time

import numpy as np

import bandplan
from cwscan import (DEFAULT_DWELL_S, DEFAULT_THRESHOLD_DB, MERGE_HZ,
                    averaged_psd, find_candidates, fmt_freq, noise_floor_db,
                    open_sdr, read_iq, scan_steps, usable_band)

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

SAMPLE_RATE = 2_048_000         # ~920 kHz usable per step.
FFT_SIZE = 32768                # 62.5 Hz bins at SAMPLE_RATE.
MEDIAN_HZ = 40_000              # noise-floor window, wider than an FM signal.
WIDTH_DROP_DB = 20.0            # width is measured down to this far below
                                # the peak (or 3 dB above the floor).
WIDTH_GAP_HZ = 500.0            # dips narrower than this don't end a signal.
# What each tuner chip can tune, by librtlsdr tuner type code.
TUNERS = {
    1: ("E4000", [(52e6, 1100e6), (1250e6, 2200e6)]),
    2: ("FC0012", [(22e6, 948.6e6)]),
    3: ("FC0013", [(22e6, 1100e6)]),
    4: ("FC2580", [(146e6, 308e6), (438e6, 924e6)]),
    5: ("R820T", [(24e6, 1766e6)]),
    6: ("R828D", [(24e6, 1766e6)]),
}
R828D = 6
# HF below the tuner, through direct sampling (RTL-SDR Blog V3) or the
# V4's built-in upconverter (R828D tuner).
HF_RANGE = (0.5e6, 28.8e6)
SPUR_STEP_HZ = 14.4e6           # half the dongle's 28.8 MHz crystal.
SPUR_GUARD_HZ = 5000.0          # strong spurs have skirts a few kHz wide.
BIRDIES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "birdies.txt")


# --------------------------------------------------------------------------
# DSP (no hardware needed)
# --------------------------------------------------------------------------

def measure_signals(freqs_hz, psd_db, threshold_db,
                    median_hz=MEDIAN_HZ):
    """Find signals in one spectrum and measure how wide each one is.

    Returns dicts with relative freq_hz, snr_db and width_hz, strongest
    first. A wide signal (SSB, FM) often breaks into several candidates;
    candidates inside an already-measured signal are dropped. The reported
    frequency is the peak for narrow signals and the middle for wide ones.
    """
    bin_hz = freqs_hz[1] - freqs_hz[0]
    floor = noise_floor_db(psd_db, int(median_hz / bin_hz) | 1)
    snr = psd_db - floor
    gap = max(1, int(WIDTH_GAP_HZ / bin_hz))
    n = len(psd_db)
    signals = []
    for cand in find_candidates(freqs_hz, psd_db, threshold_db, floor=floor):
        peak = int(np.searchsorted(freqs_hz, cand["freq_hz"]))
        if any(s["lo"] <= peak <= s["hi"] for s in signals):
            continue
        level = max(3.0, cand["snr_db"] - WIDTH_DROP_DB)
        lo = hi = peak
        for step in (-1, 1):
            i, miss = peak, 0
            while 0 <= i + step < n and miss < gap:
                i += step
                if snr[i] > level:
                    miss = 0
                    lo, hi = min(lo, i), max(hi, i)
                else:
                    miss += 1
        width = (hi - lo + 1) * bin_hz
        if width < bandplan.NARROW_HZ:
            freq = cand["freq_hz"]
        else:
            freq = (freqs_hz[lo] + freqs_hz[hi]) / 2.0
        signals.append({"freq_hz": float(freq), "snr_db": cand["snr_db"],
                        "width_hz": float(width), "lo": lo, "hi": hi})
    for s in signals:
        del s["lo"], s["hi"]
    return signals


def merge_signals(hits):
    """Merge hits of the same signal seen by two neighbouring steps.

    Hits are dicts with absolute freq_hz, snr_db and width_hz. Two hits are
    the same signal when they are closer than half the wider one's width
    (or MERGE_HZ for narrow ones); the stronger is kept.
    """
    merged = []
    for hit in sorted(hits, key=lambda h: h["freq_hz"]):
        if merged:
            last = merged[-1]
            reach = max(MERGE_HZ, max(last["width_hz"], hit["width_hz"]) / 2)
            if hit["freq_hz"] - last["freq_hz"] < reach:
                if hit["snr_db"] > last["snr_db"]:
                    merged[-1] = hit
                continue
        merged.append(hit)
    return merged


def is_crystal_harmonic(freq_hz, ppm=0):
    """True if freq_hz is near a harmonic of 14.4 MHz (half the crystal)."""
    step = SPUR_STEP_HZ * (1 + ppm * 1e-6)
    n = round(freq_hz / step)
    return n > 0 and abs(freq_hz - n * step) < SPUR_GUARD_HZ


def spur_kind(freq_hz, birdies=(), ppm=0):
    """'crystal' or 'birdie' if freq_hz is a known spur, else None."""
    if is_crystal_harmonic(freq_hz, ppm):
        return "crystal"
    if any(abs(freq_hz - b) < SPUR_GUARD_HZ for b in birdies):
        return "birdie"
    return None


def load_birdies(path):
    """Frequencies (Hz) from a birdies file; [] if it doesn't exist."""
    try:
        with open(path) as f:
            lines = [line.split("#")[0].strip() for line in f]
    except FileNotFoundError:
        return []
    return [float(line) for line in lines if line]


def save_birdies(path, hits):
    with open(path, "w") as f:
        f.write("# Birdies heard with the antenna disconnected "
                "(hamscan.py --calibrate).\n# One frequency in Hz per line.\n")
        for hit in hits:
            f.write("%.0f\n" % hit["freq_hz"])


def describe(hit, birdies=(), ppm=0):
    """Add band, likely mode, license class and spur kind to a hit."""
    freq = hit["freq_hz"]
    mode = bandplan.classify(freq, hit["width_hz"])
    hit = dict(hit, band=bandplan.band_for(freq), mode=mode,
               license=bandplan.min_class(freq, mode),
               spur=spur_kind(freq, birdies, ppm))
    if hit["spur"]:
        hit["mode"] = "spur (%s)" % hit["spur"]
    return hit


def fmt_width(hz):
    return "%.1f kHz" % (hz / 1e3) if hz >= 1000 else "%d Hz" % hz


def fmt_row(hit):
    return "%-6s %-14s %+6.1f dB  %-9s %-34s %s" % (
        hit["band"], fmt_freq(hit["freq_hz"]), hit["snr_db"],
        fmt_width(hit["width_hz"]), hit["mode"], hit["license"])


HEADER = "%-6s %-14s %9s  %-9s %-34s %s" % (
    "Band", "Frequency", "Strength", "Width", "Likely mode", "License")


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

def scan_range(sdr, start, stop, args, n_samples, birdies):
    """Sweep start..stop. Returns described hits, in frequency order."""
    low, high = usable_band(SAMPLE_RATE)
    hits = []
    for centre in scan_steps(start, stop, SAMPLE_RATE):
        sdr.center_freq = centre
        time.sleep(0.1)  # let the tuner and AGC settle
        sdr.read_samples(65536)  # discard samples from mid-retune
        iq = read_iq(sdr, n_samples)
        freqs, psd = averaged_psd(iq, SAMPLE_RATE, FFT_SIZE)
        for sig in measure_signals(freqs, psd, args.threshold):
            if not low <= sig["freq_hz"] < high:
                continue  # DC spike or band edge
            absolute = centre + sig["freq_hz"]
            if start <= absolute <= stop:
                hits.append(dict(sig, freq_hz=absolute))
    hits = [describe(h, birdies, args.ppm) for h in merge_signals(hits)]
    # Drop anything outside the amateur allocations (e.g. between the
    # 60 m channels).
    return [h for h in hits if h["license"]]


def coverage(tuner, direct_sampling="off"):
    """Frequency ranges this dongle can receive, as (low, high, ds).

    ds is the direct-sampling mode to use there: 0 (off), 1 (I) or 2 (Q).
    HF is added below the tuner's lowest frequency when direct sampling is
    on or the tuner is an R828D (V4 upconverter). Unknown tuners get an
    empty list.
    """
    ranges = [(lo, hi, 0) for lo, hi in TUNERS.get(tuner, ("", []))[1]]
    ds = {"off": 0, "i": 1, "q": 2}[direct_sampling]
    if ranges and (ds or tuner == R828D):
        hf_top = min(HF_RANGE[1], ranges[0][0])
        ranges.insert(0, (HF_RANGE[0], hf_top, ds))
    return ranges


def reachable_pieces(band, ranges):
    """The parts of a band inside the coverage ranges, as (low, high, ds)."""
    _, start, stop = band
    pieces = []
    for lo, hi, ds in ranges:
        low, high = max(start, lo), min(stop, hi)
        if low < high:
            pieces.append((low, high, ds))
    return pieces


def select_bands(spec):
    if spec == "all":
        return list(bandplan.BANDS)
    wanted = [b.strip().lower() for b in spec.split(",")]
    by_name = {b[0].lower(): b for b in bandplan.BANDS}
    unknown = [w for w in wanted if w not in by_name]
    if unknown:
        raise SystemExit("Unknown band(s): %s. Try --list-bands."
                         % ", ".join(unknown))
    return [by_name[w] for w in wanted]


def cmd_scan(args):
    bands = select_bands(args.bands)
    n_samples = max(FFT_SIZE, int(args.dwell * SAMPLE_RATE)
                    // FFT_SIZE * FFT_SIZE)
    # While calibrating, nothing is hidden: every hit is a birdie.
    birdies = [] if args.calibrate else load_birdies(args.birdies)
    if args.calibrate:
        print("Calibrating: the antenna should be disconnected, so that "
              "everything heard is the dongle's own spurs.")
    elif birdies:
        print("Hiding %d known birdie(s) from %s." % (len(birdies),
                                                      args.birdies))
    args.sample_rate = SAMPLE_RATE
    sdr = open_sdr(args)
    tuner = sdr.get_tuner_type()
    ranges = coverage(tuner, args.direct_sampling)
    print("Dongle tuner: %s, covering %s."
          % (tuner_name(tuner), fmt_coverage(ranges)))
    found = []
    hidden = 0
    unreachable = []
    try:
        for band in bands:
            name = band[0]
            pieces = reachable_pieces(band, ranges)
            if not pieces:
                unreachable.append(name)
                continue
            for start, stop, ds in pieces:
                sdr.direct_sampling = ds
                steps = len(scan_steps(start, stop, SAMPLE_RATE))
                print("%-6s %s-%s, %d step(s) ..."
                      % (name, fmt_freq(start), fmt_freq(stop), steps),
                      end="", flush=True)
                hits = scan_range(sdr, start, stop, args, n_samples, birdies)
                if not (args.show_spurs or args.calibrate):
                    hidden += sum(1 for h in hits if h["spur"])
                    hits = [h for h in hits if not h["spur"]]
                print(" %d signal(s)" % len(hits))
                for hit in hits:
                    print("  " + fmt_row(hit))
                found.extend(hits)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        sdr.close()
    print()
    if unreachable:
        print("Not scanned, outside this dongle's range: %s."
              % ", ".join(unreachable))
    if args.calibrate:
        save_birdies(args.birdies, found)
        print("Saved %d birdie(s) to %s." % (len(found), args.birdies))
        return
    if hidden:
        print("%d known spur(s) hidden; --show-spurs lists them." % hidden)
    if not found:
        print("No signals above %.0f dB." % args.threshold)
        return
    print("%d signal(s) found:" % len(found))
    print(HEADER)
    for hit in found:
        print(fmt_row(hit))


def tuner_name(tuner):
    return TUNERS.get(tuner, ("unknown tuner (type %s)" % tuner,))[0]


def fmt_short(hz):
    """Compact frequency for band edges: 135.7 kHz, 50 MHz, 122.25 GHz."""
    for unit, scale in (("GHz", 1e9), ("MHz", 1e6), ("kHz", 1e3)):
        if hz >= scale:
            return "%g %s" % (hz / scale, unit)
    return "%g Hz" % hz


def fmt_coverage(ranges):
    if not ranges:
        return "nothing known"
    return ", ".join("%s-%s%s" % (fmt_short(lo), fmt_short(hi),
                                  " (direct sampling)" if ds else "")
                     for lo, hi, ds in ranges)


def reach_label(band, ranges):
    """'yes', 'partly' or 'no' for whether coverage includes the band."""
    pieces = reachable_pieces(band, ranges)
    if not pieces:
        return "no"
    _, start, stop = band
    covered = sum(hi - lo for lo, hi, _ in pieces)
    return "yes" if covered >= stop - start else "partly"


def cmd_list_bands(args):
    """Every band with its edges and lowest license class, plus whether
    the plugged-in dongle (if any) can receive it."""
    ranges = None
    try:
        args.sample_rate = SAMPLE_RATE
        sdr = open_sdr(args)
    except SystemExit as exc:
        print("(No dongle checked: %s)\n" % str(exc).splitlines()[0])
    else:
        tuner = sdr.get_tuner_type()
        sdr.close()
        ranges = coverage(tuner, args.direct_sampling)
        print("Dongle tuner: %s, covering %s.\n"
              % (tuner_name(tuner), fmt_coverage(ranges)))
    print("%-7s %-24s %-22s %s" % ("Band", "Range", "Lowest license",
                                   "This dongle" if ranges is not None else ""))
    for band in bandplan.BANDS:
        name, start, stop = band
        print("%-7s %-24s %-22s %s" % (
            name, "%s - %s" % (fmt_short(start), fmt_short(stop)),
            bandplan.band_class(name),
            reach_label(band, ranges) if ranges is not None else ""))
    if ranges and ranges[0][0] > HF_RANGE[0]:
        print("\nHF below %s needs --direct-sampling (RTL-SDR Blog V3), a V4, "
              "or an upconverter." % fmt_short(ranges[0][0]))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="hamscan.py",
        description="Sweep the US amateur bands and list signals with band, "
                    "likely mode and license class.")
    parser.add_argument("--bands", default="all",
                        help="comma-separated bands, e.g. 2m,70cm "
                             "(default %(default)s)")
    parser.add_argument("--list-bands", action="store_true",
                        help="list the bands and exit")
    parser.add_argument("--threshold", type=float,
                        default=DEFAULT_THRESHOLD_DB,
                        help="dB above the local floor to report "
                             "(default %(default)s)")
    parser.add_argument("--dwell", type=float, default=DEFAULT_DWELL_S,
                        help="seconds averaged per step (default %(default)s)")
    parser.add_argument("--gain", default="auto",
                        help="'auto' or a gain in dB (default %(default)s)")
    parser.add_argument("--ppm", type=int, default=0,
                        help="frequency correction in ppm (default %(default)s)")
    parser.add_argument("--calibrate", action="store_true",
                        help="scan with the antenna disconnected and save "
                             "every hit as a birdie")
    parser.add_argument("--birdies", default=BIRDIES_FILE,
                        help="birdies file to read (or write with "
                             "--calibrate) (default %(default)s)")
    parser.add_argument("--show-spurs", action="store_true",
                        help="list known spurs instead of hiding them")
    parser.add_argument("--direct-sampling", choices=["off", "i", "q"],
                        default="off",
                        help="direct sampling for bands below 24 MHz on an "
                             "RTL-SDR Blog V3: off, i or q "
                             "(default %(default)s)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list_bands:
        cmd_list_bands(args)
    else:
        cmd_scan(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
