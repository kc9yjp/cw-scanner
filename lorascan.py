#!/usr/bin/env python3
"""lorascan.py — find LoRa transmissions with an RTL-SDR and list their facts.

    python3 lorascan.py                    # sweep the whole US 902-928 MHz band
    python3 lorascan.py --passes 0         # ... over and over until Ctrl-C
    python3 lorascan.py --center 904.6M    # watch one ~1.9 MHz window
    python3 lorascan.py --meshtastic       # watch Meshtastic's US default
    python3 lorascan.py --meshcore         # watch MeshCore's US default
    python3 lorascan.py --list-channels    # the region's rules and channels

For each transmission it prints: time (UTC), centre frequency, bandwidth,
spreading factor, chirp direction (up = normal, down = inverted I/Q as
LoRaWAN downlinks use), airtime, symbols, SNR, whether the airtime is
within the region's dwell limit, and the channel(s) it sits on
(LoRaWAN, Meshtastic).

Sweep or watch
--------------
LoRa packets are short and sparse, so where you listen matters more
than how widely. A sweep visits the band in ~1.4 MHz steps (19 for the
US) for --dwell seconds each, so it samples the whole band but catches
only a fraction of the traffic. --center stays on one window and listens
without gaps: the US LoRaWAN sub-band 2 that most networks use (Helium,
TTN: 903.9-905.3 MHz, plus 904.6 MHz at 500 kHz) fits in one window
centred on 904.6 MHz, and Meshtastic's default LongFast channel is
906.875 MHz.

Detection works down to about -4 dB SNR (0 dB at SF7); LoRa itself
works to -20 dB, so distant transmitters that a LoRa receiver can decode
may be missed.

Rules for other countries go in loraregions.py; pick one with --region.
--save-iq writes each packet's samples (complex64 .npy, moved to 0 Hz,
with a .json of its facts) for later decoding.
"""

import argparse
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

import lora
import loraregions
from cwscan import fmt_freq, open_sdr, parse_freq
from hamscan import TUNERS, tuner_name

SAMPLE_RATE = lora.SAMPLE_RATE
USABLE_HZ = 0.40 * SAMPLE_RATE      # trust +-960 kHz around the centre.
MAX_BW_HZ = max(lora.LORA_BANDWIDTHS)
STEP_HZ = 2 * USABLE_HZ - MAX_BW_HZ  # sweep steps overlap by one 500 kHz
                                     # channel, so no signal is ever split.
READ_BLOCK = 262144                 # samples per USB transfer (multiple of 256).
SETTLE_S = 0.05                     # discarded after tuning.
SEGMENT_S = 2.0                     # watch mode analyses this much at a time.
MAX_CARRY_S = 4.0                   # longest packet carried across segments.
SAVE_MARGIN_S = 0.005               # extra samples saved each side of a packet.
DEFAULT_DWELL_S = 2.0


# --------------------------------------------------------------------------
# Facts (no hardware needed)
# --------------------------------------------------------------------------

def describe(burst, centre_hz, t0, region):
    """Turn a burst from lora.analyse() into a fact sheet with absolute
    frequency, wall-clock time, channel names and the dwell check."""
    freq = centre_hz + burst["freq_hz"]
    bw = burst["bw_hz"]
    facts = dict(burst, freq_hz=float(freq), offset_hz=burst["freq_hz"],
                 time=t0 + burst["start_s"], region=region.code)
    facts["channels"] = (region.channels(freq, bw)
                         if bw and burst["is_lora"] else [])
    facts["dwell"] = (region.dwell_check(burst["duration_s"], bw,
                                         bool(burst["cut_off"]))
                      if burst["is_lora"] else "")
    return facts


def owned(burst, step_half_hz):
    """Whether this capture should report the burst: the whole signal fits
    in the trusted span, and its centre is in this step's share of the
    sweep (step_half_hz None: all of it)."""
    off, half = burst["freq_hz"], burst["width_hz"] / 2
    if abs(off) + half > USABLE_HZ + lora._BIN_HZ * 4:
        return False
    return step_half_hz is None or -step_half_hz <= off < step_half_hz


def sweep_centres(low_hz, high_hz):
    """Tuning frequencies whose shares (+-STEP_HZ/2 or less) tile the band."""
    n = max(1, int(np.ceil((high_hz - low_hz) / STEP_HZ)))
    step = (high_hz - low_hz) / n
    return [low_hz + step * (i + 0.5) for i in range(n)], step / 2


def fmt_time(t):
    dt = datetime.fromtimestamp(t, timezone.utc)
    return dt.strftime("%H:%M:%S.") + "%d" % (dt.microsecond // 100000)


def fmt_row(f):
    if f["bw_hz"]:
        bw = "%g kHz" % (f["bw_hz"] / 1e3)
    else:
        bw = "~%d kHz" % round(f["width_hz"] / 1e3)
    more = ">" if f["cut_off"] else ""
    airtime = "%s%.0f ms" % (more, f["duration_s"] * 1e3)
    if f["is_lora"]:
        sf = "SF%d" % f["sf"]
        chirp = "up" if f["chirp"] == "up" else "down/inv"
        symbols = "%s%.0f" % (more, f["symbols"])
    else:
        sf, chirp, symbols = "-", "none", "-"
    channel = " / ".join(f["channels"]) or ("not LoRa" if not f["is_lora"]
                                            else "-")
    return "%-10s  %-12s  %-8s %-5s %-9s %-9s %-7s %+6.1f dB  %-10s %s" % (
        fmt_time(f["time"]), fmt_freq(f["freq_hz"]), bw, sf, chirp,
        airtime, symbols, f["snr_db"], f["dwell"], channel)


HEADER = "%-10s  %-12s  %-8s %-5s %-9s %-9s %-7s %9s  %-10s %s" % (
    "Time UTC", "Frequency", "BW", "SF", "Chirps", "Airtime", "Symbols",
    "SNR", "Dwell", "Channel")


def to_jsonable(facts):
    out = {}
    for key, value in facts.items():
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def save_iq(directory, iq, facts, sample_rate):
    """Write a packet's samples (moved to 0 Hz) and its facts. Returns the
    .npy path."""
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.fromtimestamp(facts["time"], timezone.utc).strftime(
        "%Y%m%dT%H%M%S.%f")[:-3]
    base = os.path.join(directory, "lora_%sZ_%.4fMHz" % (stamp,
                                                         facts["freq_hz"] / 1e6))
    margin = int(SAVE_MARGIN_S * sample_rate)
    a = max(0, facts["start"] - margin)
    b = min(len(iq), facts["stop"] + margin)
    t = np.arange(a, b) / sample_rate
    piece = (iq[a:b] * np.exp(-2j * np.pi * facts["offset_hz"] * t))
    np.save(base + ".npy", piece.astype(np.complex64))
    meta = to_jsonable(dict(facts, sample_rate=sample_rate, first_sample=a,
                            iq_file=os.path.basename(base + ".npy")))
    with open(base + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    return base + ".npy"


# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------

class Stream:
    """Gapless samples from the dongle: read_samples_async in a thread,
    blocks handed over through a queue as (arrival time, samples)."""

    def __init__(self, sdr):
        self.sdr = sdr
        self.blocks = queue.Queue(maxsize=256)
        self.dropped = 0
        self.rest = (None, np.zeros(0, np.complex64))
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        def callback(samples, _context):
            try:
                self.blocks.put_nowait((time.time(),
                                        np.asarray(samples, np.complex64)))
            except queue.Full:
                self.dropped += len(samples)
        self.sdr.read_samples_async(callback, READ_BLOCK)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.sdr.cancel_read_async()
        self.thread.join(timeout=2)

    def read(self, n_samples):
        """(start time, samples): the next n_samples, gapless."""
        t_start, rest = self.rest
        got, size = [rest], len(rest)
        while size < n_samples:
            arrived, block = self.blocks.get(timeout=5)
            if t_start is None:
                t_start = arrived - len(block) / SAMPLE_RATE
            got.append(block)
            size += len(block)
        iq = np.concatenate(got)
        # Whatever is left over starts the next read.
        self.rest = (t_start + n_samples / SAMPLE_RATE, iq[n_samples:])
        return t_start, iq[:n_samples]


def capture(sdr, n_samples):
    """Tune-and-grab for a sweep step: (start time, samples), with the
    settling samples after the retune thrown away."""
    settle = int(SETTLE_S * SAMPLE_RATE)
    with Stream(sdr) as stream:
        t, iq = stream.read(n_samples + settle)
    return t + settle / SAMPLE_RATE, iq[settle:]


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

class Reporter:
    def __init__(self, args, region):
        self.args = args
        self.region = region
        self.lora = 0
        self.hidden = 0
        self.printed_header = False

    def report(self, facts, iq):
        if not facts["is_lora"] and not self.args.all:
            self.hidden += 1
            return
        if not self.printed_header:
            print(HEADER)
            self.printed_header = True
        self.lora += facts["is_lora"]
        line = fmt_row(facts)
        if self.args.save_iq and facts["is_lora"]:
            path = save_iq(self.args.save_iq, iq, facts, SAMPLE_RATE)
            line += "  [%s]" % os.path.basename(path)
        print(line, flush=True)

    def summary(self):
        print()
        print("%d LoRa transmission(s) found." % self.lora)
        if self.hidden:
            print("%d other wideband burst(s) (no LoRa chirps) hidden; "
                  "--all lists them." % self.hidden)


def run_sweep(sdr, args, region, reporter):
    low, high = region.low_hz, region.high_hz
    if args.start or args.stop:
        low = parse_freq(args.start) if args.start else low
        high = parse_freq(args.stop) if args.stop else high
    centres, half = sweep_centres(low, high)
    n_samples = int(args.dwell * SAMPLE_RATE) // READ_BLOCK * READ_BLOCK
    n_samples = max(n_samples, READ_BLOCK)
    print("Sweeping %s-%s in %d step(s) of %.1f s; Ctrl-C stops."
          % (fmt_freq(low), fmt_freq(high), len(centres), args.dwell))
    done = 0
    while args.passes == 0 or done < args.passes:
        for centre in centres:
            sdr.center_freq = centre
            t0, iq = capture(sdr, n_samples)
            for burst in lora.analyse(iq, SAMPLE_RATE, args.threshold):
                if owned(burst, half):
                    facts = describe(burst, centre, t0, region)
                    if low <= facts["freq_hz"] <= high:
                        reporter.report(facts, iq)
        done += 1


def run_watch(sdr, args, region, reporter):
    """Listen on one window without gaps, in SEGMENT_S pieces. A packet
    still going at the end of a piece is carried into the next one, so
    it is measured whole."""
    centre = parse_freq(args.center)
    sdr.center_freq = centre
    if args.preset:
        p = args.preset
        print("%s US default: %s, %s kHz, SF%d."
              % (p.name, fmt_freq(p.freq_hz), "%g" % (p.bw_hz / 1e3), p.sf))
    seg = int(SEGMENT_S * SAMPLE_RATE)
    print("Watching %s-%s (centre %s); Ctrl-C stops."
          % (fmt_freq(centre - USABLE_HZ), fmt_freq(centre + USABLE_HZ),
             fmt_freq(centre)))
    deadline = time.time() + args.duration if args.duration else None
    with Stream(sdr) as stream:
        stream.read(int(SETTLE_S * SAMPLE_RATE))
        carry, t_carry, recent = np.zeros(0, np.complex64), None, []
        while deadline is None or time.time() < deadline:
            t_new, fresh = stream.read(seg)
            t0 = t_carry if len(carry) else t_new
            iq = np.concatenate([carry, fresh])
            carry, t_carry, recent = watch_segment(
                iq, t0, centre, args, region, reporter, recent)
        if stream.dropped:
            print("Warning: %d samples dropped (computer too slow)."
                  % stream.dropped)


def watch_segment(iq, t0, centre, args, region, reporter, recent=()):
    """Analyse one piece and report what has finished.

    Returns (samples to carry over, their start time, reported): a packet
    still going at the end is carried into the next piece from its start.
    Packets reported now that lie in the carried samples will be seen
    again, so `reported` lists them as (time, freq, width) and passing it
    back as `recent` skips them next time.
    """
    finished, keep_from = [], len(iq)
    max_carry = int(MAX_CARRY_S * SAMPLE_RATE)
    for burst in lora.analyse(iq, SAMPLE_RATE, args.threshold):
        if not owned(burst, None):
            continue
        ends_late = burst["cut_off"] in ("end", "both")
        if ends_late and len(iq) - burst["start"] < max_carry:
            keep_from = min(keep_from, max(0, burst["start"] - lora.FRAME * 8))
        else:
            finished.append(describe(burst, centre, t0, region))
    reported = []
    for facts in finished:
        if any(abs(facts["time"] - t) < 0.02
               and abs(facts["freq_hz"] - f) < w / 4 for t, f, w in recent):
            continue
        reporter.report(facts, iq)
        reported.append((facts["time"], facts["freq_hz"], facts["width_hz"]))
    return iq[keep_from:], t0 + keep_from / SAMPLE_RATE, reported


def check_tuner(sdr, region):
    tuner = sdr.get_tuner_type()
    ranges = TUNERS.get(tuner, ("", []))[1]
    ok = any(lo <= region.low_hz and region.high_hz <= hi for lo, hi in ranges)
    print("Dongle tuner: %s.%s" % (
        tuner_name(tuner), "" if ok else
        " Warning: it may not tune all of %s-%s."
        % (fmt_freq(region.low_hz), fmt_freq(region.high_hz))))


def cmd_scan(args):
    region = loraregions.REGIONS[args.region]
    args.sample_rate = SAMPLE_RATE
    args.direct_sampling = "off"
    sdr = open_sdr(args)
    reporter = Reporter(args, region)
    print("Region: %s." % region.name)
    try:
        check_tuner(sdr, region)
        if args.center:
            run_watch(sdr, args, region, reporter)
        else:
            run_sweep(sdr, args, region, reporter)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sdr.close()
    reporter.summary()


def cmd_list_channels(args):
    region = loraregions.REGIONS[args.region]
    print("%s" % region.name)
    print("  Band:       %s - %s" % (fmt_freq(region.low_hz),
                                     fmt_freq(region.high_hz)))
    print("  Rules:      %s" % region.rules)
    print("  Max power:  %s" % region.max_power)
    if region.dwell_limit_s:
        exempt = (" (%d kHz and wider: no limit)"
                  % (region.dwell_exempt_bw_hz / 1e3)
                  if region.dwell_exempt_bw_hz else "")
        print("  Dwell:      %.0f ms per transmission%s"
              % (region.dwell_limit_s * 1e3, exempt))
    print()
    print("%-18s %-8s %-9s %-28s %s" % ("Plan", "BW", "Channels", "Range",
                                        "Spacing"))
    for plan in region.plans:
        last = plan.first_number + plan.count - 1
        print("%-18s %-8s %-9s %-28s %s" % (
            plan.name, "%g kHz" % (plan.bw_hz / 1e3),
            "%d-%d" % (plan.first_number, last),
            "%s - %s" % (fmt_freq(plan.centre(plan.first_number)),
                         fmt_freq(plan.centre(last))),
            "%g kHz" % (plan.spacing_hz / 1e3)))
        for number, name in sorted(plan.named.items()):
            print("    %d = %s, %s" % (number, name,
                                       fmt_freq(plan.centre(number))))
    if region.presets:
        print()
        print("Mesh defaults (watch with --<key>):")
        for key, p in region.presets.items():
            print("  --%-11s %-20s %s, %g kHz, SF%d" % (
                key, p.name, fmt_freq(p.freq_hz), p.bw_hz / 1e3, p.sf))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="lorascan.py",
        description="Find LoRa transmissions and list their basic facts.")
    parser.add_argument("--region", default=loraregions.DEFAULT_REGION,
                        choices=sorted(loraregions.REGIONS),
                        help="regulatory region (default %(default)s)")
    parser.add_argument("--list-channels", action="store_true",
                        help="show the region's rules and channel plans, "
                             "then exit")
    parser.add_argument("--center",
                        help="watch one window centred here (e.g. 904.6M) "
                             "instead of sweeping")
    nets = parser.add_mutually_exclusive_group()
    nets.add_argument("--meshtastic", action="store_true",
                      help="watch Meshtastic's default channel (US: "
                           "LongFast, 906.875 MHz)")
    nets.add_argument("--meshcore", action="store_true",
                      help="watch MeshCore's default channel (US: "
                           "910.525 MHz)")
    parser.add_argument("--duration", type=float, default=0,
                        help="with --center: stop after this many seconds "
                             "(default: until Ctrl-C)")
    parser.add_argument("--start", help="sweep from here (default: band edge)")
    parser.add_argument("--stop", help="sweep to here (default: band edge)")
    parser.add_argument("--dwell", type=float, default=DEFAULT_DWELL_S,
                        help="seconds per sweep step (default %(default)s)")
    parser.add_argument("--passes", type=int, default=1,
                        help="sweeps to make; 0 = until Ctrl-C "
                             "(default %(default)s)")
    parser.add_argument("--threshold", type=float,
                        default=lora.DEFAULT_THRESHOLD_DB,
                        help="detection threshold, dB over the noise in a "
                             "125 kHz x 4 ms box (default %(default)s)")
    parser.add_argument("--all", action="store_true",
                        help="also list wideband bursts that aren't LoRa")
    parser.add_argument("--save-iq", metavar="DIR",
                        help="save each LoRa packet's samples and facts here")
    parser.add_argument("--gain", default="auto",
                        help="'auto' or a gain in dB (default %(default)s)")
    parser.add_argument("--ppm", type=int, default=0,
                        help="frequency correction in ppm (default %(default)s)")
    return parser


def apply_preset(args, parser):
    """--meshtastic / --meshcore: watch centred on the region's default
    channel for that network."""
    args.preset = None
    chosen = [k for k in ("meshtastic", "meshcore") if getattr(args, k)]
    if not chosen:
        return
    if args.center:
        parser.error("--%s and --center can't be used together" % chosen[0])
    region = loraregions.REGIONS[args.region]
    preset = region.presets.get(chosen[0])
    if preset is None:
        parser.error("region %s has no %s default" % (region.code, chosen[0]))
    args.preset = preset
    args.center = "%.0f" % preset.freq_hz


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_preset(args, parser)
    if args.list_channels:
        cmd_list_channels(args)
    else:
        cmd_scan(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
