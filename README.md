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
- **`lorascan.py`** — finds LoRa transmissions in the 902-928 MHz band
  (US rules for now) and lists each one's facts: time, frequency,
  bandwidth, spreading factor, chirp direction, airtime, SNR, whether it
  keeps to the FCC dwell limit, and the LoRaWAN or Meshtastic channel it
  is on. It doesn't decode the payload yet.

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

### LoRa scanner

```sh
# The US rules and channel plans it knows
python3 lorascan.py --list-channels

# Sweep 902-928 MHz once (19 steps of 2 s, about a minute)
python3 lorascan.py

# Keep sweeping until Ctrl-C
python3 lorascan.py --passes 0

# Watch LoRaWAN sub-band 2 (Helium, TTN) without gaps
python3 lorascan.py --center 904.6M

# Watch the US default channel of Meshtastic (LongFast, 906.875 MHz)
# or MeshCore (910.525 MHz), here for 10 minutes
python3 lorascan.py --meshtastic --duration 600
python3 lorascan.py --meshcore --duration 600

# Save each packet's samples for decoding later
python3 lorascan.py --center 904.6M --save-iq captures/
```

Example output (from the synthetic self-test):

```
Time UTC    Frequency     BW       SF    Chirps    Airtime   Symbols       SNR  Dwell      Channel
14:02:11.2  904.3000 MHz  125 kHz  SF9   up        172 ms    42        +5.5 dB  ok         LoRaWAN uplink 10 (sub-band 2) / Meshtastic slot 19
14:02:15.0  923.2984 MHz  500 kHz  SF8   down/inv  21 ms     42        +2.9 dB  n/a        LoRaWAN downlink 0
```

- **BW / SF** — the occupied bandwidth (62.5, 125, 250 or 500 kHz) is measured
  off the spectrum. The spreading factor (SF7-12) is found by
  "dechirping" the preamble at each SF and keeping the one that turns
  into a clean tone.
- **Chirps** — `up` is normal LoRa. `down/inv` means inverted I/Q, which
  LoRaWAN gateways use for downlinks, so it says which way the packet
  was going.
- **Airtime / Symbols** — how long it was on the air, and how many
  symbols that is at its SF and bandwidth. A `>` means the packet was
  already going when the capture started, or still going when it ended.
- **Dwell** — FCC 15.247 allows frequency-hopping systems narrower than
  500 kHz to stay on one channel for at most 400 ms; 500 kHz signals have
  no limit (`n/a`).
- **Channel** — every channel plan the frequency fits. The grids
  overlap: Meshtastic's 125 kHz slots sit 12.5 kHz from many LoRaWAN
  uplinks, so some packets match both.

Only LoRa is listed. The band is full of other short bursts (smart
meters and other frequency hoppers, mostly 60-90 kHz FSK); `--all` lists
those too, as `not LoRa`.

**Sweep or watch.** LoRa packets are short and sparse. A sweep samples the
whole band but listens to each 1.4 MHz piece only 2 s at a time, so it
misses most traffic. `--center` stays on one ~1.9 MHz window and listens
without gaps, which is better once you know where to look.

**Sensitivity.** Packets are found down to about -4 dB SNR at SF8-12, and
about 0 dB at SF7 (which has the least spreading gain). LoRa itself
works down to -20 dB, so a LoRa radio can decode distant transmitters
that this scanner misses.

**Mesh defaults.** `--meshtastic` and `--meshcore` are shortcuts for
`--center` on each network's default US channel: Meshtastic LongFast is
906.875 MHz, 250 kHz, SF11; MeshCore's recommended US/Canada preset
(since October 2025) is 910.525 MHz, 62.5 kHz, SF7. `--list-channels`
shows them. Each region keeps its own defaults in `loraregions.py`.

**Frequency correction.** Channel names allow only ~8-25 kHz of error,
and a dongle without a TCXO can be off by more. The test R820T read
906.875 MHz Meshtastic packets as 906.826 MHz, 49 kHz low, until it was
run with `--ppm 54`. If a strong packet shows no channel name, find your
dongle's ppm the same way: watch a known channel and adjust `--ppm` until
it lands.

**Other countries.** Each region's rules (band, dwell limit, power, channel
plans) are data in `loraregions.py`. Only `US915` is filled in. To add
another country, add a `Region` there and it becomes available as
`--region`.

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
- `lorascan.py` — the LoRa scanner: tuning, sweep and watch modes, the
  fact table, `--save-iq`.
- `lora.py` — LoRa DSP with no hardware: burst detection, bandwidth
  measurement, SF and chirp direction by dechirping. This is where a
  decoder will go.
- `loraregions.py` — LoRa rules and channel plans per region (US915).
- `test_synth.py`, `test_hamscan.py`, `test_lora.py` — self-tests with
  synthetic signals. Run them any time:
  ```sh
  python3 test_synth.py
  python3 test_hamscan.py
  python3 test_lora.py
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

`lorascan.py` is checked on synthetic packets (every SF at 125 kHz, plus
250 and 500 kHz, inverted I/Q, -4 dB SNR, two channels at once, and FSK
and too-short chirp bursts that must not count) and on a fake dongle for
the sweep and the gapless watch stream (17 tests). On the R820T dongle it
has swept the whole band and watched sub-band 2, the Meshtastic LongFast
channel and the MeshCore channel. A first version called a few short FSK bursts LoRa; the
chirp threshold was set from ~70 real off-air bursts (the highest scored
~9 dB, LoRa scores 14+), and bursts shorter than 10 symbols no longer
count. With `--meshtastic --ppm 54` it heard real Meshtastic LongFast
packets (SF11, 250 kHz, about 600 ms, +40 dB from a nearby node) and
named the channel correctly. Nothing was heard on the MeshCore channel
during the test.

## What's next

- **LoRa decoding** — use the samples saved with `--save-iq` to find the
  sync word (LoRaWAN or private/Meshtastic), then demodulate, de-whiten
  and decode the header and payload. LoRaWAN headers (device address,
  frame counter) and Meshtastic headers are readable without keys.

- **Web UI** — run the scanner on a Linux box and listen/control it from a
  phone browser, the way MultiScan's web control works.
- **Logging** — record what was heard, when, and on what frequency (a CSV
  log per day, like MultiScan's).
- **Multi-VFO** — listen to several signals at once instead of one.
- **Better decoding** — the current decoder assumes clean keying; fading
  and interference will need adaptive thresholds and error tolerance.
- **CW band plans** — scan only the CW sub-bands by default, instead of
  arbitrary segments.
