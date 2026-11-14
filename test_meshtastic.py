#!/usr/bin/env python3
"""test_meshtastic.py — self-test for meshtastic.py and aes.py.

No hardware or network needed.

    python3 test_meshtastic.py

Exits 0 when every test passes, 1 otherwise.
"""

import base64
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aes
import meshtastic as mt


def test_aes_vectors():
    """FIPS-197 appendix C: AES-128 and AES-256."""
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    ct = aes.encrypt_block(aes.expand_key(bytes(range(16))), pt)
    assert ct.hex() == "69c4e0d86a7b0430d8cdb78070b4c55a", ct.hex()
    ct = aes.encrypt_block(aes.expand_key(bytes(range(32))), pt)
    assert ct.hex() == "8ea2b7ca516745bfeafc49904b496089", ct.hex()


def test_aes_ctr():
    """NIST SP 800-38A F.5.1, CTR-AES128, first two blocks."""
    key = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    iv = bytes.fromhex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
    pt = bytes.fromhex("6bc1bee22e409f96e93d7e117393172a"
                       "ae2d8a571e03ac9c9eb76fac45af8e51")
    ct = aes.ctr(key, iv, pt)
    assert ct.hex() == ("874d6191b620e3261bef6864990db6ce"
                        "9806f66b7970fdff8617187bb9fffdff"), ct.hex()
    assert aes.ctr(key, iv, ct) == pt


def test_default_key():
    """The public default key is 'AQ==' expanded, and the well-known
    LongFast channel hash is 8."""
    assert mt.expand_psk(base64.b64decode("AQ==")) == mt.DEFAULT_KEY
    assert base64.b64encode(mt.DEFAULT_KEY) == b"1PG7OiApB1nwvP+rz05pAQ=="
    assert mt.channel_hash("LongFast", mt.DEFAULT_KEY) == 8
    assert mt.expand_psk(b"\x00") is None
    assert mt.expand_psk(b"\x02")[-1] == mt.DEFAULT_KEY[-1] + 1
    name, key = mt.parse_psk("Secret:" + base64.b64encode(bytes(32)).decode())
    assert name == "Secret" and key == bytes(32)


def test_protobuf():
    """Wire format: varints, length-delimited, fixed32; garbage rejected."""
    buf = bytes([0x08, 0x96, 0x01, 0x12, 0x03]) + b"abc" + bytes([0x25]) \
        + struct.pack("<I", 7)
    f = mt.parse_protobuf(buf)
    assert f == {1: [150], 2: [b"abc"], 4: [struct.pack("<I", 7)]}, f
    for bad in (bytes([0x00, 0x01]), bytes([0x0B, 0x01]), bytes([0x12, 0x09, 1])):
        try:
            mt.parse_protobuf(bad)
        except ValueError:
            continue
        raise AssertionError("accepted %s" % bad.hex())


def test_text_round_trip():
    """A LongFast text message encrypts and decodes back."""
    pkt = mt.encode(mt.BROADCAST, 0xA1B2C3D4, 0x1234ABCD, 1, b"hello mesh",
                    hop_limit=2, hop_start=3)
    m = mt.decode(pkt)
    assert m["channel"] == "LongFast" and not m["encrypted"], m
    assert m["content"] == {"text": "hello mesh"}, m
    assert (m["sender"], m["dest"], m["id"]) == (0xA1B2C3D4, mt.BROADCAST,
                                                 0x1234ABCD)
    assert (m["hop_limit"], m["hop_start"]) == (2, 3)
    line = mt.summary(m)
    for text in ("!a1b2c3d4 -> ^all", "1234abcd", "hops 2/3", "[LongFast]",
                 'TEXT_MESSAGE: "hello mesh"'):
        assert text in line, (text, line)
    assert "relayed" not in line, line
    relayed = mt.decode(mt.encode(mt.BROADCAST, 0xA1B2C3D4, 7, 1, b"x",
                                  hop_limit=2, relay=0xF0))
    assert "relayed by !......f0" in mt.summary(relayed), mt.summary(relayed)


def test_position_nodeinfo_telemetry():
    """Position, node info and telemetry bodies are read."""
    pos = (bytes([0x0D]) + struct.pack("<i", 377749000)
           + bytes([0x15]) + struct.pack("<i", -1224194000)
           + bytes([0x18, 0x0A]))
    m = mt.decode(mt.encode(mt.BROADCAST, 1, 2, 3, pos))
    c = m["content"]
    assert abs(c["lat"] - 37.7749) < 1e-6 and abs(c["lon"] + 122.4194) < 1e-6
    assert c["alt_m"] == 10
    user = (bytes([0x0A, 9]) + b"!00000001" + bytes([0x12, 4]) + b"Base"
            + bytes([0x1A, 2]) + b"BS")
    m = mt.decode(mt.encode(mt.BROADCAST, 1, 3, 4, user))
    assert m["content"]["long_name"] == "Base" and "Base" in mt.summary(m)
    dev = bytes([0x08, 87]) + bytes([0x15]) + struct.pack("<f", 4.1)
    tele = bytes([0x12, len(dev)]) + dev
    m = mt.decode(mt.encode(mt.BROADCAST, 1, 4, 67, tele))
    assert m["content"]["battery"] == 87 and "battery 87%" in mt.summary(m)


def test_wrong_and_custom_keys():
    """A packet on a private channel stays encrypted unless its key is
    given, and then decodes, named."""
    key = bytes(range(16))
    pkt = mt.encode(mt.BROADCAST, 5, 6, 1, b"secret", key=key, channel="Ops")
    m = mt.decode(pkt)
    assert m["encrypted"] and m["channel"] is None, m
    assert "encrypted, no matching key" in mt.summary(m)
    m = mt.decode(pkt, [("Ops", key)])
    assert m["content"]["text"] == "secret" and m["channel"] == "Ops", m
    m = mt.decode(pkt, [(None, key)])   # key without a name still works
    assert m["content"]["text"] == "secret", m


def test_pki_direct_message():
    """Channel hash 0 to a single node is a public-key DM: flagged, not
    decrypted."""
    pkt = struct.pack("<III", 0x11111111, 0x22222222, 9) + bytes(
        [3, 0, 0, 0]) + bytes(20)
    m = mt.decode(pkt)
    assert m["pki"] and "public-key" in mt.summary(m), m


def test_short_payload():
    assert mt.decode(b"\x01\x02") is None


TESTS = [
    ("AES FIPS-197 vectors", test_aes_vectors),
    ("AES-CTR NIST vector", test_aes_ctr),
    ("default key and LongFast hash", test_default_key),
    ("protobuf wire format", test_protobuf),
    ("text message round trip", test_text_round_trip),
    ("position, node info, telemetry", test_position_nodeinfo_telemetry),
    ("private channel keys", test_wrong_and_custom_keys),
    ("public-key direct message", test_pki_direct_message),
    ("too short to be Meshtastic", test_short_payload),
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
