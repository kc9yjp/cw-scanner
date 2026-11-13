# CW Scanner

A first version of a homebrew CW (Morse code) scanner and listener for an
RTL-SDR dongle, in Python, for Linux. Built instead of using MultiScan SDR
— which is Windows/macOS only and closed source — so it runs natively on
Linux and can grow CW-specific features (Morse decoding, CW band segments,
logging) that a general scanner doesn't bother with.

## What it does

- **`scan`** — sweeps a band segment and lists the frequencies where
  signals rise above the local noise floor. Detection is deliberately
  simple: averaged FFT, rolling-median noise floor (the same idea as
  MultiScan's "pseudo-spectrum"), peaks more than 10 dB above the floor.
- **`listen`** — tunes to one frequency, saves what it hears as a WAV file
  (700 Hz audio tone), and prints a live Morse decode of it.

## Install (Arch)

```sh
./install-arch.sh
source .venv/bin/activate
```

This installs `rtl-sdr`, numpy and scipy with pacman, puts pyrtlsdr in a
`.venv` virtualenv (it isn't in the official repos), checks the dongle
setup and runs the self-test. It's safe to run again.

## Install (Debian / Ubuntu)

```sh
sudo apt install rtl-sdr librtlsdr-dev
pip install --break-system-packages pyrtlsdr numpy scipy
```

Or use a virtualenv instead of `--break-system-packages`. You also need a
dongle plugged in, with USB permission (the `rtl-sdr` package's udev rules
normally handle that; add your user to the `plugdev` group if needed).

## Usage

```sh
# Sweep the 40 m CW segment (7.000-7.040 MHz) and list signals
python3 cwscan.py scan

# Sweep a different segment, e.g. the 20 m CW segment
python3 cwscan.py scan --start 14.0M --stop 14.07M

# Listen to one frequency for 30 seconds, save audio + decode it
python3 cwscan.py listen 7030000 --duration 30

# Same, but tell the decoder the speed (helps when the signal is weak)
python3 cwscan.py listen 7.03M --duration 30 --wpm 20 --output cq.wav
```

Frequencies accept plain Hz (`7030000`), `k`/`kHz`, or `M`/`MHz`.

## HF notes — read this before wondering why 40 m is silent

Most CW lives on HF, 1.8-30 MHz, and the dongle's tuner doesn't cover HF
directly. How you get there depends on the dongle:

- **RTL-SDR Blog V4** — has a built-in upconverter, so HF just works.
  No special flags needed.
- **RTL-SDR Blog V3** — needs direct-sampling mode (the tuner is bypassed
  and the ADC samples HF directly) or an external upconverter. Use
  `--direct-sampling q` (Q-ADC is the usual choice):
  ```sh
  python3 cwscan.py scan --direct-sampling q
  python3 cwscan.py listen 7030000 --direct-sampling q
  ```
- Above ~30 MHz (2 m, 70 cm) neither flag is needed on either dongle.

If the band sounds dead, check the antenna first: a random length of wire
works far better on HF than the little telescopic whip that ships with
most dongles.

## Files

- `cwscan.py` — the CLI tool (`scan` / `listen`). All DSP is pure
  numpy/scipy; only the dongle code needs pyrtlsdr.
- `morse.py` — Morse encoder and a simple mark/space timing decoder.
  Hardware-independent, so it can be tested on its own.
- `test_synth.py` — self-test with synthetic signals. Run it any time:
  ```sh
  python3 test_synth.py
  ```

## Status: hardware testing still to be done

No RTL-SDR hardware is attached to the machine this was built on, so the
DSP and decoding logic were verified with synthetic signals only — the
self-test generates a known "CQ" in Morse, checks the detector finds it
and the decoder reads it back (7 tests, all passing). The `scan` and
`listen` paths against a real dongle are untested: the first real run is
the moment of truth for tuner settling, gain, and whether the noise floor
behaves. If the first scan is noisy or empty, try `--gain` with a manual
value instead of `auto`, and `--threshold` to raise or lower sensitivity.

## What's next

- **Web UI** — run the scanner on a Linux box and listen/control it from a
  phone browser, the way MultiScan's web control works.
- **Logging** — record what was heard, when, and on what frequency (a CSV
  log per day, like MultiScan's).
- **Multi-VFO** — listen to several signals at once instead of one.
- **Better decoding** — the current decoder assumes clean keying; fading
  and interference will need adaptive thresholds and error tolerance.
- **CW band plans** — scan only the CW sub-bands by default, instead of
  arbitrary segments.
