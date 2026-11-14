"""loraregions.py — LoRa frequency rules and channel plans, by region.

Pure data plus lookups, no hardware or DSP, so it can be tested on its own.

Each Region holds the band LoRa may use there, the regulator's limits
that can be checked from the air (how long one transmission may last),
and the channel plans in common use, so a detected signal can be named
("LoRaWAN uplink 8", "Meshtastic slot 20").

Only the US is filled in. To add a country, write another Region with
its band, limits and channel plans and add it to REGIONS; lorascan.py
picks it up with --region.
"""

from dataclasses import dataclass, field

LORA_BANDWIDTHS = (125_000, 250_000, 500_000)   # Hz, the ones used in practice.


@dataclass(frozen=True)
class ChannelPlan:
    """A grid of equally spaced channels of one bandwidth.

    Channel n (counting from `first_number`) is centred at
    first_hz + (n - first_number) * spacing_hz. `named` gives well-known
    channels a nickname, keyed by channel number.
    """
    name: str
    first_hz: float
    spacing_hz: float
    count: int
    bw_hz: float
    first_number: int = 0
    named: dict = field(default_factory=dict)
    label_fn: object = None     # optional fn(number) -> extra text

    def centre(self, number):
        return self.first_hz + (number - self.first_number) * self.spacing_hz

    def nearest(self, freq_hz):
        """(channel number, offset Hz) of the nearest channel."""
        i = round((freq_hz - self.first_hz) / self.spacing_hz)
        i = min(max(i, 0), self.count - 1)
        number = i + self.first_number
        return number, freq_hz - self.centre(number)

    def label(self, number):
        text = "%s %d" % (self.name, number)
        extra = [self.label_fn(number)] if self.label_fn else []
        if number in self.named:
            extra.append(self.named[number])
        return text + (" (%s)" % ", ".join(extra) if extra else "")


@dataclass(frozen=True)
class Region:
    code: str                   # what --region takes, e.g. "US915"
    name: str
    low_hz: float               # band LoRa may use
    high_hz: float
    rules: str                  # the regulation, for display
    max_power: str              # for display; can't be measured off the air
    dwell_limit_s: float        # longest single transmission allowed for
    dwell_exempt_bw_hz: float   # signals narrower than this (None: no limit)
    plans: tuple = ()

    def channels(self, freq_hz, bw_hz=None, tolerance_hz=None):
        """Labels of every channel the signal could be on, in plan order
        (the plans are listed most common first).

        Only plans of the signal's bandwidth are tried when bw_hz is
        given. The default tolerance is an eighth of the channel
        bandwidth, at most 25 kHz (15.6 kHz at 125 kHz): enough for a
        dongle a dozen ppm off frequency. Grids overlap -- Meshtastic's
        125 kHz slots sit 12.5 kHz from many LoRaWAN uplinks -- so a
        signal can match more than one.
        """
        hits = []
        for plan in self.plans:
            if bw_hz is not None and plan.bw_hz != bw_hz:
                continue
            tol = (tolerance_hz if tolerance_hz is not None
                   else min(plan.bw_hz / 8, 25_000))
            number, offset = plan.nearest(freq_hz)
            if abs(offset) <= tol:
                hits.append(plan.label(number))
        return hits

    def dwell_check(self, duration_s, bw_hz, cut_off=False):
        """'ok', 'over limit', or 'n/a' (no limit at this bandwidth).

        cut_off means the transmission was still going when the capture
        started or ended, so duration_s is only a lower bound.
        """
        if self.dwell_limit_s is None or (
                self.dwell_exempt_bw_hz and bw_hz
                and bw_hz >= self.dwell_exempt_bw_hz):
            return "n/a"
        if duration_s > self.dwell_limit_s:
            return "over limit"
        return "ok?" if cut_off else "ok"

    def in_band(self, freq_hz):
        return self.low_hz <= freq_hz <= self.high_hz


def _k(khz):
    return khz * 1e3


def _m(mhz):
    return mhz * 1e6


def _meshtastic(bw_hz, low_hz, high_hz, named=None):
    """Meshtastic's slot grid: slots of one bandwidth packed edge to edge
    from the bottom of the band, numbered from 1."""
    return ChannelPlan("Meshtastic slot", low_hz + bw_hz / 2, bw_hz,
                       int((high_hz - low_hz) // bw_hz), bw_hz,
                       first_number=1, named=named or {})


# --------------------------------------------------------------------------
# United States: FCC Part 15.247, 902-928 MHz ISM band.
#
# * LoRaWAN US915 (LoRa Alliance RP002): 64 uplink channels of 125 kHz
#   from 902.3 MHz every 200 kHz, in 8 sub-bands of 8 (most networks,
#   Helium and TTN included, use sub-band 2: channels 8-15 plus 65);
#   8 uplink channels of 500 kHz from 903.0 MHz every 1.6 MHz; 8 downlink
#   channels of 500 kHz from 923.3 MHz every 600 kHz.
# * Meshtastic: slots of the preset's bandwidth from 902.0 MHz. The
#   default LongFast channel is slot 20 of the 250 kHz grid, 906.875 MHz.
# * Dwell: frequency-hopping systems narrower than 500 kHz may stay on
#   one channel at most 400 ms at a time. 500 kHz signals count as
#   digital modulation (6 dB width >= 500 kHz) and have no dwell limit.
# --------------------------------------------------------------------------

US915 = Region(
    code="US915",
    name="United States (FCC Part 15.247, 902-928 MHz)",
    low_hz=_m(902), high_hz=_m(928),
    rules="FCC 47 CFR 15.247",
    max_power="1 W (30 dBm) conducted, 4 W (36 dBm) EIRP with a 6 dBi antenna",
    dwell_limit_s=0.400,
    dwell_exempt_bw_hz=500_000,
    plans=(
        ChannelPlan("LoRaWAN uplink", _m(902.3), _k(200), 64, 125_000,
                    label_fn=lambda n: "sub-band %d" % (n // 8 + 1)),
        ChannelPlan("LoRaWAN uplink", _m(903.0), _k(1600), 8, 500_000,
                    first_number=64,
                    label_fn=lambda n: "sub-band %d" % (n - 64 + 1)),
        ChannelPlan("LoRaWAN downlink", _m(923.3), _k(600), 8, 500_000),
        _meshtastic(125_000, _m(902), _m(928)),
        _meshtastic(250_000, _m(902), _m(928),
                    named={20: "LongFast default"}),
        _meshtastic(500_000, _m(902), _m(928)),
    ),
)

REGIONS = {r.code: r for r in (US915,)}
DEFAULT_REGION = "US915"
