"""Static encrypted storage of hidden evaluation data (standard library only).

The hidden set exists on disk only as ciphertext; plaintext is decrypted only during evaluation into the judge's temporary directory and deleted afterwards.
The agent stage never gets the key (the runner neither puts the key file in the agent view nor writes it into the agent's environment variables).

Format: b"ABHID1" | nonce(16) | ciphertext | tag(32)
  keystream_i = SHA256(enc_key || nonce || i.to_bytes(8))      (SHA-256 counter mode)
  tag         = HMAC-SHA256(mac_key, b"ABHID1" || nonce || ciphertext)
  enc_key / mac_key are derived from the master key via HMAC.

This is a stopgap without third-party crypto libraries; once the container backend is ready it can switch to AES-GCM, with the header version bumped.

Usage:
  python3 tools/hidden_store.py keygen  KEYFILE
  python3 tools/hidden_store.py encrypt KEYFILE PLAIN  OUT.enc
  python3 tools/hidden_store.py decrypt KEYFILE IN.enc OUT
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
from pathlib import Path

MAGIC = b"ABHID1"


def _keys(master: bytes) -> tuple[bytes, bytes]:
    return (hmac.new(master, b"abench-hidden-enc", hashlib.sha256).digest(),
            hmac.new(master, b"abench-hidden-mac", hashlib.sha256).digest())


def _xor_stream(enc_key: bytes, nonce: bytes, data: bytes) -> bytes:
    out = bytearray(len(data))
    for block in range(0, len(data), 32):
        ks = hashlib.sha256(enc_key + nonce + (block // 32).to_bytes(8, "big")).digest()
        chunk = data[block:block + 32]
        out[block:block + len(chunk)] = bytes(a ^ b for a, b in zip(chunk, ks))
    return bytes(out)


def read_key(path: Path) -> bytes:
    raw = Path(path).read_text(encoding="ascii").strip()
    key = bytes.fromhex(raw)
    if len(key) != 32:
        raise ValueError("hidden key must be 32 bytes hex")
    return key


def encrypt_bytes(master: bytes, plain: bytes) -> bytes:
    enc_key, mac_key = _keys(master)
    nonce = secrets.token_bytes(16)
    ct = _xor_stream(enc_key, nonce, plain)
    tag = hmac.new(mac_key, MAGIC + nonce + ct, hashlib.sha256).digest()
    return MAGIC + nonce + ct + tag


def decrypt_bytes(master: bytes, blob: bytes) -> bytes:
    if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + 16 + 32:
        raise ValueError("not an ABench hidden-store file")
    enc_key, mac_key = _keys(master)
    nonce = blob[len(MAGIC):len(MAGIC) + 16]
    ct, tag = blob[len(MAGIC) + 16:-32], blob[-32:]
    if not hmac.compare_digest(tag, hmac.new(mac_key, MAGIC + nonce + ct, hashlib.sha256).digest()):
        raise ValueError("hidden-store integrity check failed (wrong key or modified file)")
    return _xor_stream(enc_key, nonce, ct)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__); return 2
    cmd = argv[1]
    if cmd == "keygen" and len(argv) == 3:
        p = Path(argv[2]); p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32) + "\n")
        print("wrote", p); return 0
    if cmd in ("encrypt", "decrypt") and len(argv) == 5:
        key = read_key(Path(argv[2]))
        data = Path(argv[3]).read_bytes()
        out = encrypt_bytes(key, data) if cmd == "encrypt" else decrypt_bytes(key, data)
        Path(argv[4]).write_bytes(out)
        print("wrote", argv[4], len(out), "bytes"); return 0
    print(__doc__); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
