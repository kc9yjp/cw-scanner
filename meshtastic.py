"""meshtastic.py — read Meshtastic packets from their LoRa payload.

Pure Python, no hardware. lorademod.decode() gives the payload bytes;
decode() here splits off the 16-byte Meshtastic header, works out the
channel from its hash, decrypts with the channel key (AES-CTR) and
decodes the protobuf inside: text messages, positions, node info,
telemetry, and the names of everything else.

Packet layout (firmware 2.x)
----------------------------
  0  dest     uint32 LE   0xFFFFFFFF = broadcast
  4  sender   uint32 LE   node number, shown as !xxxxxxxx
  8  id       uint32 LE   packet id
 12  flags    hop_limit (bits 0-2), want_ack (3), via_mqtt (4),
              hop_start (bits 5-7)
 13  channel  hash of the channel name and key
 14  next_hop, 15 relay_node: last byte of a node number (0 = none);
              a relayed copy carries the relaying node's
 16  payload  encrypted meshtastic.Data protobuf

Encryption is AES-CTR with the channel key and a nonce of packet id
(8 bytes LE) + sender (4 bytes LE) + 4 zero bytes. The default channel
key ("AQ==", expanded below) is public, so default-channel traffic is
readable by any Meshtastic node, and by this. Channels with their own
key, and direct messages encrypted to a node's public key (channel hash
0), stay unreadable without the key.
"""

import base64
import struct

import aes

DEFAULT_KEY = bytes.fromhex("d4f1bb3a20290759f0bcffabcf4e6901")
BROADCAST = 0xFFFFFFFF
HEADER_LEN = 16

# Modem presets: the name an unnamed channel goes by, and so hashes with.
PRESETS = ("LongFast", "LongSlow", "LongModerate", "LongTurbo",
           "VeryLongSlow", "MediumSlow", "MediumFast", "ShortSlow",
           "ShortFast", "ShortTurbo")

PORTNUMS = {
    0: "UNKNOWN", 1: "TEXT_MESSAGE", 2: "REMOTE_HARDWARE", 3: "POSITION",
    4: "NODEINFO", 5: "ROUTING", 6: "ADMIN", 7: "TEXT_MESSAGE_COMPRESSED",
    8: "WAYPOINT", 9: "AUDIO", 10: "DETECTION_SENSOR", 11: "ALERT",
    32: "REPLY", 33: "IP_TUNNEL", 34: "PAXCOUNTER", 64: "SERIAL",
    65: "STORE_FORWARD", 66: "RANGE_TEST", 67: "TELEMETRY", 68: "ZPS",
    69: "SIMULATOR", 70: "TRACEROUTE", 71: "NEIGHBORINFO",
    72: "ATAK_PLUGIN", 73: "MAP_REPORT", 74: "POWERSTRESS",
    76: "RETICULUM_TUNNEL", 77: "CAYENNE", 256: "PRIVATE",
    257: "ATAK_FORWARDER",
}


# --------------------------------------------------------------------------
# Keys and channels
# --------------------------------------------------------------------------

def expand_psk(psk):
    """A channel PSK as stored by Meshtastic -> AES key (or None for no
    encryption). One byte n means the default key with its last byte
    increased by n - 1 (n = 0: no encryption); 16 or 32 bytes are used
    as they are."""
    psk = bytes(psk)
    if len(psk) == 0 or psk == b"\x00":
        return None
    if len(psk) == 1:
        return DEFAULT_KEY[:-1] + bytes([(DEFAULT_KEY[-1] + psk[0] - 1) & 0xFF])
    if len(psk) in (16, 32):
        return psk
    raise ValueError("a PSK is 0, 1, 16 or 32 bytes, not %d" % len(psk))


def parse_psk(text):
    """'NAME:BASE64' or 'BASE64' -> (name or None, AES key or None)."""
    name, _, b64 = text.rpartition(":")
    return (name or None), expand_psk(base64.b64decode(b64))


def channel_hash(name, key):
    """The one-byte channel id: XOR of the name's bytes and the key's."""
    h = 0
    for b in name.encode() + (key or b""):
        h ^= b
    return h


def default_channels():
    """(name, key) of every preset channel with the default key."""
    return [(name, DEFAULT_KEY) for name in PRESETS]


# --------------------------------------------------------------------------
# Protobuf (just the wire format)
# --------------------------------------------------------------------------

def _varint(buf, i):
    value, shift = 0, 0
    while True:
        if i >= len(buf) or shift > 63:
            raise ValueError("bad varint")
        b = buf[i]
        value |= (b & 0x7F) << shift
        i += 1
        shift += 7
        if not b & 0x80:
            return value, i


def parse_protobuf(buf):
    """Fields of a protobuf message as {field number: [values]}: ints for
    varints, bytes for length-delimited, raw 4/8 bytes for fixed. Raises
    ValueError on anything malformed (which is how a wrong key shows)."""
    fields, i = {}, 0
    buf = bytes(buf)
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ValueError("field 0")
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 2:
            length, i = _varint(buf, i)
            value, i = buf[i:i + length], i + length
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError("wire type %d" % wire)
        if i > len(buf):
            raise ValueError("truncated")
        fields.setdefault(number, []).append(value)
    return fields


def _first(fields, number, default=None):
    return fields.get(number, [default])[0]


def _signed32(value):
    return value - (1 << 32) if value >= 1 << 31 else value


def _float(raw):
    return struct.unpack("<f", raw)[0] if isinstance(raw, bytes) and len(raw) == 4 else None


def _fixed32(raw, signed=False):
    if not isinstance(raw, bytes) or len(raw) != 4:
        return None
    return struct.unpack("<i" if signed else "<I", raw)[0]


def _text(raw):
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else None


# --------------------------------------------------------------------------
# Packet
# --------------------------------------------------------------------------

def node_id(num):
    return "^all" if num == BROADCAST else "!%08x" % num


def parse_header(payload):
    """The 16-byte header as a dict, or None if the payload is shorter."""
    if len(payload) < HEADER_LEN:
        return None
    dest, sender, packet_id = struct.unpack_from("<III", payload)
    flags, chan, next_hop, relay = payload[12:16]
    return {"dest": dest, "sender": sender, "id": packet_id,
            "hop_limit": flags & 7, "want_ack": bool(flags & 8),
            "via_mqtt": bool(flags & 16), "hop_start": flags >> 5,
            "channel_hash": chan, "next_hop": next_hop, "relay": relay}


def decrypt(key, packet_id, sender, data):
    nonce = struct.pack("<QI", packet_id, sender) + bytes(4)
    return aes.ctr(key, nonce, data)


def parse_data(plain):
    """A decrypted meshtastic.Data message -> dict, or None if it isn't
    one (wrong key)."""
    try:
        f = parse_protobuf(plain)
    except ValueError:
        return None
    port = _first(f, 1)
    if not isinstance(port, int) or port not in PORTNUMS:
        return None
    body = _first(f, 2, b"")
    if not isinstance(body, bytes):
        return None
    return {"portnum": port, "port": PORTNUMS[port], "body": body,
            "request_id": _fixed32(_first(f, 6)),
            "reply_id": _fixed32(_first(f, 7))}


def decode_body(port, body):
    """What's inside, for the common apps, as a dict (empty if unknown or
    unparseable)."""
    try:
        if port == 1:
            return {"text": body.decode("utf-8", "replace")}
        f = parse_protobuf(body)
        if port == 3:       # Position
            out = {}
            lat, lon = _fixed32(_first(f, 1), True), _fixed32(_first(f, 2), True)
            if lat is not None and lon is not None:
                out["lat"], out["lon"] = lat * 1e-7, lon * 1e-7
            if 3 in f:
                out["alt_m"] = _signed32(_first(f, 3))
            if 4 in f:
                out["time"] = _fixed32(_first(f, 4))
            return out
        if port == 4:       # User (node info)
            return {"id": _text(_first(f, 1)), "long_name": _text(_first(f, 2)),
                    "short_name": _text(_first(f, 3)),
                    "hw_model": _first(f, 5), "role": _first(f, 7)}
        if port == 67:      # Telemetry
            out = {}
            if 2 in f:
                d = parse_protobuf(_first(f, 2))
                out.update(battery=_first(d, 1), voltage=_float(_first(d, 2)),
                           ch_util=_float(_first(d, 3)),
                           air_util_tx=_float(_first(d, 4)),
                           uptime_s=_first(d, 5))
            if 3 in f:
                e = parse_protobuf(_first(f, 3))
                out.update(temp_c=_float(_first(e, 1)),
                           humidity=_float(_first(e, 2)),
                           pressure_hpa=_float(_first(e, 3)))
            return {k: v for k, v in out.items() if v is not None}
        if port == 70:      # RouteDiscovery
            return {"route": [node_id(_fixed32(r)) for r in f.get(1, [])
                              if isinstance(r, bytes) and len(r) == 4]}
    except (ValueError, UnicodeDecodeError, struct.error):
        pass
    return {}


def decode(payload, channels=None):
    """Read a Meshtastic packet. channels: (name, key) pairs to try, in
    addition to the default-key presets.

    Returns a dict: the header fields, channel (name or None),
    encrypted (True when no key fitted), pki (direct message to a public
    key: unreadable by design), and when decrypted: portnum, port,
    content (see decode_body) and body (raw bytes). None if the payload
    is too short to be Meshtastic.
    """
    head = parse_header(payload)
    if head is None:
        return None
    out = dict(head, channel=None, encrypted=True, pki=False,
               port=None, portnum=None, content={}, body=b"")
    data = bytes(payload[HEADER_LEN:])
    if head["channel_hash"] == 0 and head["dest"] != BROADCAST:
        out["pki"] = True
        return out
    tried = list(channels or []) + default_channels()
    for name, key in tried:
        if key is None:
            plain = data
        elif name is not None and channel_hash(name, key) != head["channel_hash"]:
            continue
        else:
            plain = decrypt(key, head["id"], head["sender"], data)
        parsed = parse_data(plain)
        if parsed:
            out.update(channel=name, encrypted=False, port=parsed["port"],
                       portnum=parsed["portnum"], body=parsed["body"],
                       content=decode_body(parsed["portnum"], parsed["body"]))
            return out
    return out


def encode(dest, sender, packet_id, portnum, body, key=DEFAULT_KEY,
           channel="LongFast", hop_limit=3, hop_start=3, want_ack=False,
           relay=None):
    """Build a Meshtastic packet payload (for tests)."""
    data = bytes([0x08]) + _enc_varint(portnum) + bytes([0x12]) \
        + _enc_varint(len(body)) + bytes(body)
    flags = (hop_limit & 7) | (8 if want_ack else 0) | ((hop_start & 7) << 5)
    relay = sender & 0xFF if relay is None else relay
    head = struct.pack("<III", dest, sender, packet_id) + bytes(
        [flags, channel_hash(channel, key), 0, relay])
    return head + decrypt(key, packet_id, sender, data)


def _enc_varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def summary(m):
    """One line describing a decoded packet."""
    hops = "hops %d/%d" % (m["hop_limit"], m["hop_start"]) \
        if m["hop_start"] else "hops %d" % m["hop_limit"]
    head = "%s -> %s  id %08x  %s" % (node_id(m["sender"]), node_id(m["dest"]),
                                      m["id"], hops)
    if m["relay"] and m["relay"] != m["sender"] & 0xFF:
        # Only the last byte of the relaying node's number is sent.
        head += "  relayed by !......%02x" % m["relay"]
    if m["via_mqtt"]:
        head += "  via MQTT"
    if m["pki"]:
        return head + "  direct message (public-key encrypted)"
    if m["encrypted"]:
        return head + "  channel 0x%02x: encrypted, no matching key" \
            % m["channel_hash"]
    c = m["content"]
    what = m["port"]
    if "text" in c:
        what += ': "%s"' % c["text"]
    elif m["portnum"] == 3 and "lat" in c:
        what += " %.5f, %.5f" % (c["lat"], c["lon"])
        if "alt_m" in c:
            what += " %d m" % c["alt_m"]
    elif m["portnum"] == 4 and c.get("long_name"):
        what += ' %s "%s" (%s)' % (c.get("id") or "", c["long_name"],
                                   c.get("short_name") or "")
    elif m["portnum"] == 67 and c:
        parts = []
        if "battery" in c:
            parts.append("battery %d%%" % c["battery"])
        if "voltage" in c:
            parts.append("%.2f V" % c["voltage"])
        if "temp_c" in c:
            parts.append("%.1f C" % c["temp_c"])
        if "humidity" in c:
            parts.append("%.0f%% RH" % c["humidity"])
        if "ch_util" in c:
            parts.append("ch util %.1f%%" % c["ch_util"])
        what += " " + ", ".join(parts)
    elif m["portnum"] == 70 and c.get("route") is not None:
        what += " via " + (" ".join(c["route"]) or "direct")
    else:
        what += " (%d bytes)" % len(m["body"])
    return "%s  [%s]  %s" % (head, m["channel"], what)
