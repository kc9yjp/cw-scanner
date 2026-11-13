"""bandplan.py — US amateur bands, license privileges and band-plan usage.

Pure data plus lookups, no hardware or DSP, so it can be tested on its own.

Three tables, all in Hz:

* PRIVILEGES -- FCC Part 97.301 segments: who may transmit where. Each
  segment gives the lowest license class allowed to send CW there and the
  lowest allowed to send anything else. On 80, 40 and 15 m, Technicians
  have CW-only slices; on 50.0-50.1 and 144.0-144.1 MHz only CW is
  allowed at all.
* USAGE -- the ARRL band plan, simplified to one usage per segment (CW,
  CW/data, phone, weak-signal, beacon, FM, satellite, ATV, mixed). This is
  convention, not law, but it is what makes a mode guess possible.
* DIGITAL -- the well-known FT8, FT4 and WSPR watering holes, as
  (dial frequency, low offset, high offset): signals sit that far above
  the USB dial frequency.

Classify a detected signal with band_for(), classify() and min_class().
"""

CLASSES = ("Technician", "General", "Extra")
T, G, E = CLASSES

# Approximate width classes for a signal measured off the spectrum (to
# 20 dB below its peak, so skirts count: SSB measures ~4-4.2 kHz, NBFM
# voice ~12-14 kHz).
NARROW_HZ = 500       # CW, carriers, beacons, single digital tones.
VOICE_HZ = 5_000      # SSB and narrow data (RTTY, PSK, Olivia ...).
NBFM_HZ = 25_000      # AM, FM and digital voice. Anything wider is wide.


def k(khz):
    return khz * 1e3


def m(mhz):
    return mhz * 1e6


# (name, low Hz, high Hz), in frequency order.
BANDS = [
    ("2200m", k(135.7), k(137.8)),
    ("630m", k(472), k(479)),
    ("160m", k(1800), k(2000)),
    ("80m", k(3500), k(4000)),
    ("60m", k(5330), k(5407)),
    ("40m", k(7000), k(7300)),
    ("30m", k(10100), k(10150)),
    ("20m", k(14000), k(14350)),
    ("17m", k(18068), k(18168)),
    ("15m", k(21000), k(21450)),
    ("12m", k(24890), k(24990)),
    ("10m", k(28000), k(29700)),
    ("6m", m(50), m(54)),
    ("2m", m(144), m(148)),
    ("219MHz", m(219), m(220)),
    ("1.25m", m(222), m(225)),
    ("70cm", m(420), m(450)),
    ("33cm", m(902), m(928)),
    ("23cm", m(1240), m(1300)),
    ("13cm", m(2300), m(2450)),      # two pieces, see PRIVILEGES
    # 9 cm (3.3-3.5 GHz) is left out: the FCC voted in 2020 to remove the
    # amateur allocation there.
    ("5cm", m(5650), m(5925)),
    ("3cm", m(10000), m(10500)),
    ("1.2cm", m(24000), m(24250)),
    ("6mm", m(47000), m(47200)),
    ("4mm", m(76000), m(81000)),
    ("2.5mm", m(122250), m(123000)),
    ("2mm", m(134000), m(141000)),
    ("1mm", m(241000), m(250000)),
]

# (low Hz, high Hz, lowest class for CW, lowest class for other modes).
# None means the mode isn't allowed there at all.
PRIVILEGES = [
    (k(135.7), k(137.8), G, G),
    (k(472), k(479), G, G),
    (k(1800), k(2000), G, G),
    (k(3500), k(3525), E, E),
    (k(3525), k(3600), T, G),        # Technician: CW only
    (k(3600), k(3800), E, E),
    (k(3800), k(4000), G, G),
    # 60 m is five 2.8 kHz channels, given by centre frequency.
    (k(5332 - 1.4), k(5332 + 1.4), G, G),
    (k(5348 - 1.4), k(5348 + 1.4), G, G),
    (k(5358.5 - 1.4), k(5358.5 + 1.4), G, G),
    (k(5373 - 1.4), k(5373 + 1.4), G, G),
    (k(5405 - 1.4), k(5405 + 1.4), G, G),
    (k(7000), k(7025), E, E),
    (k(7025), k(7125), T, G),        # Technician: CW only
    (k(7125), k(7175), E, E),
    (k(7175), k(7300), G, G),
    (k(10100), k(10150), G, G),
    (k(14000), k(14025), E, E),
    (k(14025), k(14150), G, G),
    (k(14150), k(14225), E, E),
    (k(14225), k(14350), G, G),
    (k(18068), k(18168), G, G),
    (k(21000), k(21025), E, E),
    (k(21025), k(21200), T, G),      # Technician: CW only
    (k(21200), k(21275), E, E),
    (k(21275), k(21450), G, G),
    (k(24890), k(24990), G, G),
    (k(28000), k(28500), T, T),      # Technician: CW, data, SSB to 28.5
    (k(28500), k(29700), G, G),
    (m(50.0), m(50.1), T, None),     # CW only, all classes
    (m(50.1), m(54), T, T),
    (m(144.0), m(144.1), T, None),   # CW only, all classes
    (m(144.1), m(148), T, T),
    (m(219), m(220), None, T),       # fixed digital message forwarding only
    (m(222), m(225), T, T),
    (m(420), m(450), T, T),
    (m(902), m(928), T, T),
    (m(1240), m(1300), T, T),
    (m(2300), m(2310), T, T),
    (m(2390), m(2450), T, T),
    (m(5650), m(5925), T, T),
    (m(10000), m(10500), T, T),
    (m(24000), m(24250), T, T),
    (m(47000), m(47200), T, T),
    (m(76000), m(81000), T, T),
    (m(122250), m(123000), T, T),
    (m(134000), m(141000), T, T),
    (m(241000), m(250000), T, T),
]

# (low Hz, high Hz, usage). Usages: cw, cwdata, phone, weak (VHF+ weak
# signal CW/SSB), beacon, fm (FM repeaters and simplex), sat, atv, mixed.
USAGE = [
    (k(135.7), k(137.8), "cwdata"),
    (k(472), k(479), "cwdata"),
    (k(1800), k(1843), "cwdata"),
    (k(1843), k(2000), "phone"),
    (k(3500), k(3600), "cwdata"),
    (k(3600), k(4000), "phone"),
    (k(5330), k(5407), "phone"),     # 60 m channels: USB, CW and data
    (k(7000), k(7125), "cwdata"),
    (k(7125), k(7300), "phone"),
    (k(10100), k(10150), "cwdata"),
    (k(14000), k(14150), "cwdata"),
    (k(14150), k(14350), "phone"),
    (k(18068), k(18110), "cwdata"),
    (k(18110), k(18168), "phone"),
    (k(21000), k(21200), "cwdata"),
    (k(21200), k(21450), "phone"),
    (k(24890), k(24930), "cwdata"),
    (k(24930), k(24990), "phone"),
    (k(28000), k(28200), "cwdata"),
    (k(28200), k(28300), "beacon"),
    (k(28300), k(29300), "phone"),
    (k(29300), k(29510), "sat"),
    (k(29510), k(29700), "fm"),
    (m(50.0), m(50.1), "cw"),
    (m(50.1), m(50.3), "weak"),
    (m(50.3), m(51.0), "mixed"),
    (m(51.0), m(54.0), "fm"),
    (m(144.0), m(144.1), "cw"),
    (m(144.1), m(144.275), "weak"),
    (m(144.275), m(144.3), "beacon"),
    (m(144.3), m(144.5), "sat"),
    (m(144.5), m(145.8), "fm"),
    (m(145.8), m(146.0), "sat"),
    (m(146.0), m(148.0), "fm"),
    (m(219), m(220), "mixed"),
    (m(222.0), m(222.15), "weak"),
    (m(222.15), m(225.0), "fm"),
    (m(420), m(432), "atv"),
    (m(432.0), m(432.07), "cw"),
    (m(432.07), m(432.3), "weak"),
    (m(432.3), m(432.4), "beacon"),
    (m(432.4), m(433), "mixed"),
    (m(433), m(435), "fm"),
    (m(435), m(438), "sat"),
    (m(438), m(442), "mixed"),
    (m(442), m(450), "fm"),
    (m(902), m(903.4), "weak"),
    (m(903.4), m(928), "mixed"),
    (m(1240), m(1294), "mixed"),
    (m(1294), m(1295), "fm"),
    (m(1295), m(1296), "mixed"),
    (m(1296.0), m(1296.05), "cw"),
    (m(1296.05), m(1296.4), "weak"),
    (m(1296.4), m(1297), "beacon"),
    (m(1297), m(1300), "mixed"),
    # Microwave: the weak-signal calling areas; everything else is mixed.
    (m(2300), m(2304), "mixed"),
    (m(2304), m(2304.3), "weak"),
    (m(2304.3), m(2310), "mixed"),
    (m(2390), m(2450), "mixed"),
    (m(5650), m(5760), "mixed"),
    (m(5760), m(5760.3), "weak"),
    (m(5760.3), m(5925), "mixed"),
    (m(10000), m(10368), "mixed"),
    (m(10368), m(10368.3), "weak"),
    (m(10368.3), m(10500), "mixed"),
    (m(24000), m(24192), "mixed"),
    (m(24192), m(24192.3), "weak"),
    (m(24192.3), m(24250), "mixed"),
]

# (mode, dial Hz, low offset Hz, high offset Hz). The audio passband of an
# FT8/FT4 signal runs from about 200 Hz to 3 kHz above the dial; WSPR
# lives in a 200 Hz slot 1400-1600 Hz above it. First match wins.
_FT = (0, 3000)
_WSPR = (1400, 1600)
DIGITAL = (
    [("FT8", k(f)) + _FT for f in (1840, 3573, 5357, 7074, 10136, 14074,
                                     18100, 21074, 24915, 28074)]
    + [("FT8", m(50.313)) + _FT, ("FT8", m(144.174)) + _FT]
    + [("FT4", k(f)) + _FT for f in (3575, 7047.5, 10140, 14080, 18104,
                                     21140, 24919, 28180)]
    + [("FT4", m(50.318)) + _FT]
    + [("WSPR", k(f)) + _WSPR for f in (1836.6, 3568.6, 7038.6, 10138.7,
                                        14095.6, 18104.6, 21094.6,
                                        24924.6, 28124.6)]
    + [("WSPR", m(50.293)) + _WSPR, ("WSPR", m(144.489)) + _WSPR]
)

# Likely mode by band-plan usage and measured width:
# (narrow, voice-width, FM-width). Wider than NBFM_HZ is always wideband.
_MODES = {
    "cw":     ("CW", "unknown (narrowband)", "FM or wide data"),
    "cwdata": ("CW", "data (RTTY/PSK/etc)", "FM or wide data"),
    "phone":  ("CW or AM carrier", "SSB", "AM/FM"),
    "weak":   ("CW", "SSB", "FM"),
    "beacon": ("CW beacon", "SSB/data", "FM"),
    "fm":     ("FM carrier (no audio)", "digital voice", "FM"),
    "sat":    ("satellite beacon/CW", "SSB/data (satellite)", "FM (satellite)"),
    "atv":    ("carrier", "SSB/data", "FM/digital voice"),
    "mixed":  ("CW/carrier", "SSB/data", "FM/digital voice"),
}

CW_MODES = ("CW", "CW beacon", "CW or AM carrier", "satellite beacon/CW",
            "CW/carrier")


def band_for(freq_hz):
    """Band name for a frequency, or None if outside every band."""
    for name, low, high in BANDS:
        if low <= freq_hz <= high:
            return name
    return None


def usage_for(freq_hz):
    """Band-plan usage at a frequency, or None."""
    for low, high, usage in USAGE:
        if low <= freq_hz < high:
            return usage
    return None


def digital_for(freq_hz, width_hz):
    """FT8/FT4/WSPR if the signal is in one of their slots and no more
    than 1 kHz wider than the slot, else None."""
    for mode, dial, lo_off, hi_off in DIGITAL:
        if (dial + lo_off <= freq_hz <= dial + hi_off
                and width_hz <= hi_off - lo_off + 1000):
            return mode
    return None


def classify(freq_hz, width_hz):
    """Best guess at the mode of a signal at freq_hz, width_hz wide."""
    digital = digital_for(freq_hz, width_hz)
    if digital:
        return digital
    if width_hz >= NBFM_HZ:
        return "wideband (ATV/data)"
    narrow, voice, fm = _MODES.get(usage_for(freq_hz), _MODES["mixed"])
    if width_hz < NARROW_HZ:
        return narrow
    if width_hz < VOICE_HZ:
        return voice
    return fm


def min_class(freq_hz, mode):
    """Lowest US license class that may transmit `mode` at freq_hz.

    Returns one of CLASSES, or None when the frequency is outside the
    amateur allocations (or the mode isn't permitted there).
    """
    for low, high, cw_class, other_class in PRIVILEGES:
        if low <= freq_hz < high:
            # In a CW-only (or data-only) segment, whatever was heard,
            # report the class for the mode that is allowed there.
            if mode in CW_MODES:
                return cw_class or other_class
            return other_class or cw_class
    return None


def band_class(name):
    """Lowest license class with any privileges in the named band, with
    " (CW only)" added when that class may only send CW there (Technician
    on 80, 40 and 15 m)."""
    _, low, high = next(b for b in BANDS if b[0] == name)
    segments = [(cw, other) for lo, hi, cw, other in PRIVILEGES
                if low <= lo and hi <= high]
    classes = {c for pair in segments for c in pair if c}
    lowest = min(classes, key=CLASSES.index)
    if not any(other == lowest for _, other in segments):
        return lowest + " (CW only)"
    return lowest
