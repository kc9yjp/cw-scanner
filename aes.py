"""aes.py — AES-128/256 block encryption and CTR mode, in pure Python.

Just enough AES to decrypt Meshtastic packets (AES-CTR) without adding
a dependency. Slow (a few thousand blocks a second) but LoRa packets are
at most 16 blocks. Checked against the FIPS-197 test vectors in
test_meshtastic.py. Not for protecting anything: no constant-time care.
"""


def _xtime(a):
    a <<= 1
    return (a ^ 0x11B) if a & 0x100 else a


def _mul(a, b):
    out = 0
    while b:
        if b & 1:
            out ^= a
        a = _xtime(a)
        b >>= 1
    return out


def _sbox():
    """The S-box built from its definition (inverse in GF(2^8), then the
    affine map), rather than a typed-in table."""
    box = []
    for x in range(256):
        inv = next((y for y in range(1, 256) if _mul(x, y) == 1), 0)
        s = inv
        for shift in range(1, 5):
            s ^= ((inv << shift) | (inv >> (8 - shift))) & 0xFF
        box.append(s ^ 0x63)
    return box


SBOX = _sbox()


def expand_key(key):
    """Round keys (as 16-byte lists) for a 16- or 32-byte key."""
    nk = len(key) // 4
    if len(key) not in (16, 32):
        raise ValueError("AES key must be 16 or 32 bytes")
    rounds = nk + 6
    words = [list(key[4 * i:4 * i + 4]) for i in range(nk)]
    rcon = 1
    for i in range(nk, 4 * (rounds + 1)):
        w = list(words[i - 1])
        if i % nk == 0:
            w = w[1:] + w[:1]
            w = [SBOX[b] for b in w]
            w[0] ^= rcon
            rcon = _xtime(rcon)
        elif nk > 6 and i % nk == 4:
            w = [SBOX[b] for b in w]
        words.append([a ^ b for a, b in zip(words[i - nk], w)])
    return [sum(words[4 * r:4 * r + 4], []) for r in range(rounds + 1)]


def encrypt_block(round_keys, block):
    s = [b ^ k for b, k in zip(block, round_keys[0])]
    last = len(round_keys) - 1
    for r in range(1, last + 1):
        s = [SBOX[b] for b in s]
        # ShiftRows (state is column-major: byte i is row i % 4).
        s = [s[(i + 4 * (i % 4)) % 16] for i in range(16)]
        if r != last:
            out = []
            for c in range(4):
                a = s[4 * c:4 * c + 4]
                out += [
                    _mul(a[0], 2) ^ _mul(a[1], 3) ^ a[2] ^ a[3],
                    a[0] ^ _mul(a[1], 2) ^ _mul(a[2], 3) ^ a[3],
                    a[0] ^ a[1] ^ _mul(a[2], 2) ^ _mul(a[3], 3),
                    _mul(a[0], 3) ^ a[1] ^ a[2] ^ _mul(a[3], 2),
                ]
            s = out
        s = [b ^ k for b, k in zip(s, round_keys[r])]
    return bytes(s)


def ctr(key, nonce, data):
    """AES-CTR: XOR data with E(counter), the 16-byte counter starting at
    nonce and counting up big-endian. Encrypts and decrypts alike."""
    keys = expand_key(bytes(key))
    counter = int.from_bytes(nonce, "big")
    out = bytearray()
    for i in range(0, len(data), 16):
        stream = encrypt_block(keys, counter.to_bytes(16, "big"))
        out += bytes(a ^ b for a, b in zip(data[i:i + 16], stream))
        counter = (counter + 1) % (1 << 128)
    return bytes(out)
