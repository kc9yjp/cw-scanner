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
  MultiScan's "pseudo-spectrum"), peaks more than 5 dB above the floor.
  Each step averages 1 s of samples and skips the dongle's DC spike at the
  tuned frequency, so weak signals show up without false alarms.
- **`listen`** — tunes to one frequency, saves what it hears as a WAV file
  (700 Hz audio tone), and prints a live Morse decode of it.
- **`hamscan.py`** — sweeps the US amateur bands (2200 m to 1 mm, as far
  as the dongle reaches) and lists each signal with its band, frequency,
  strength, width, likely mode, and the lowest license class (Technician,
  General or Extra) allowed to transmit that mode there.

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

# Dig for very weak signals: average longer and lower the threshold
python3 cwscan.py scan --dwell 3 --threshold 3

# Listen to one frequency for 30 seconds, save audio + decode it
python3 cwscan.py listen 7030000 --duration 30

# Same, but tell the decoder the speed (helps when the signal is weak)
python3 cwscan.py listen 7.03M --duration 30 --wpm 20 --output cq.wav
```

Frequencies accept plain Hz (`7030000`), `k`/`kHz`, or `M`/`MHz`.

### Ham band scanner

```sh
# Every band, what this dongle can reach, and the lowest license class
python3 hamscan.py --list-bands

# Sweep every band the dongle can reach (about 4 minutes for 12 m-23 cm)
python3 hamscan.py

# Just some bands
python3 hamscan.py --bands 10m,6m,2m

# HF too, on an RTL-SDR Blog V3
python3 hamscan.py --direct-sampling q
```

Example output:

```
Band   Frequency       Strength  Width     Likely mode                        License
10m    28.7898 MHz     +11.9 dB  3.0 kHz   SSB                                General
2m     146.9984 MHz     +5.7 dB  62 Hz     FM carrier (no audio)              Technician
```

**License** is the lowest class allowed to transmit that mode on that
frequency (FCC Part 97.301), so it accounts for the Technician CW-only
slices on 80, 40 and 15 m. Only the classes issued today are used, so
the segments grandfathered Advanced licensees share with Extra (such as
7.125-7.175 MHz) show as Extra.

**Likely mode** is a guess from the measured width and the ARRL band plan
(plus the FT8/FT4/WSPR frequencies): narrow is CW or a carrier, ~3-4 kHz
is SSB or data, ~12-14 kHz is FM. An idle FM carrier looks just like CW.

**Spurs ("birdies").** A dongle hears its own clocks, and on UHF most of
the hits can be birdies. Harmonics of 14.4 MHz (which land on 144.000,
432.000 and 1296.000 MHz) are always hidden. For the rest, disconnect the
antenna once and run:

```sh
python3 hamscan.py --calibrate
```

That saves every hit to `birdies.txt`, and later scans hide them (and say
how many). `--show-spurs` lists them, marked. Recalibrate if you change
dongle, computer or USB port.

**Coverage.** The band list is every US amateur band; what's scanned
depends on the dongle. `hamscan.py` reads the tuner chip at startup
(E4000, FC0012, FC0013, FC2580, R820T or R828D), scans the part of each
band that chip can tune, and lists the bands it couldn't reach. HF below
the tuner is added with `--direct-sampling` (RTL-SDR Blog V3) or
automatically on an R828D (V4); see the HF notes below. Microwave bands
(13 cm and up) are beyond any RTL-SDR without a downconverter. 9 cm
(3.3-3.5 GHz) is left out because the FCC voted in 2020 to remove that
allocation.

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
- **Generic dongles** (EEPROM says `Realtek` / `RTL2838UHIDIR`; check with
  `rtl_eeprom`) — can't receive HF at all. The R820T tuner stops at about
  24 MHz (below that you get `[R82XX] PLL not locked!`), and
  `--direct-sampling` runs but has no antenna path wired to it, so you
  just get noise. You need a V3, a V4, or an external upconverter.
- Above ~30 MHz (2 m, 70 cm) neither flag is needed on any dongle.

If the band sounds dead, check the antenna first: a random length of wire
works far better on HF than the little telescopic whip that ships with
most dongles.

## Files

- `cwscan.py` — the CLI tool (`scan` / `listen`). All DSP is pure
  numpy/scipy; only the dongle code needs pyrtlsdr.
- `morse.py` — Morse encoder and a simple mark/space timing decoder.
  Hardware-independent, so it can be tested on its own.
- `hamscan.py` — the all-bands scanner. Reuses `cwscan.py`'s detector.
- `bandplan.py` — US band edges, license privileges, band-plan usage and
  digital-mode frequencies, with the lookups `hamscan.py` uses.
- `test_synth.py`, `test_hamscan.py` — self-tests with synthetic signals.
  Run them any time:
  ```sh
  python3 test_synth.py
  python3 test_hamscan.py
  ```

## Status

The self-test covers detection (including weak signals and the DC spike),
scan step planning, and decoding with synthetic signals (12 tests).

On real hardware, `scan` and `listen` have been run on a generic R820T
dongle at VHF: `scan` finds FM broadcast carriers and stays quiet on an
empty 2 m CW segment, and `listen` records and decodes. A real CW signal
has not been decoded off the air yet. If scans are noisy or empty, try
`--gain` with a manual value instead of `auto`, `--threshold` to raise or
lower sensitivity, and `--dwell` to average longer.

`hamscan.py` has swept 12 m to 23 cm on the same dongle. Without a
birdies file, most UHF hits were the dongle's own spurs (14.4 MHz
harmonics and a 128 kHz comb), which is why `--calibrate` exists. Its
mode guesses have been checked on synthetic CW, SSB and FM (8 tests), not
yet against signals of known mode off the air.

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
