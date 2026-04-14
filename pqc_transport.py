"""
PQC Transport Layer
====================
Wraps JSON alert/threat payloads with Post-Quantum Cryptography:
  • Kyber-512 KEM for session-key establishment
  • AES-256-GCM symmetric encryption per message
  • SHA3-256 integrity check on decrypted payloads
  • Session key reuse with per-message unique nonce
  • Automatic key rotation every N messages
  • Replay protection via monotonic sequence counter + timestamp
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pqc import KyberKEM, KyberPublicKey, KyberSecretKey, KyberCiphertext, get_kyber_params


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

KEY_ROTATION_INTERVAL: int = 1000       # rotate session key every N messages
MAX_CLOCK_SKEW: float = 30.0            # max allowed timestamp drift (seconds)
NONCE_SIZE: int = 12                    # AES-GCM nonce length


# ──────────────────────────────────────────────────────────────────────
# Public Key Serialization Helpers
# ──────────────────────────────────────────────────────────────────────

def serialize_public_key(pk: KyberPublicKey) -> dict:
    """Serialize a KyberPublicKey to a JSON-safe dict."""
    return {
        "t": [arr.tolist() for arr in pk.t],
        "rho": base64.b64encode(pk.rho).decode("ascii"),
    }


def deserialize_public_key(data: dict) -> KyberPublicKey:
    """Deserialize a KyberPublicKey from a JSON-safe dict."""
    return KyberPublicKey(
        t=[np.array(arr, dtype=np.int64) for arr in data["t"]],
        rho=base64.b64decode(data["rho"]),
    )


def serialize_ciphertext(ct: KyberCiphertext) -> dict:
    """Serialize a KyberCiphertext to a JSON-safe dict."""
    return {
        "u": [arr.tolist() for arr in ct.u],
        "v": ct.v.tolist(),
    }


def deserialize_ciphertext(data: dict) -> KyberCiphertext:
    """Deserialize a KyberCiphertext from a JSON-safe dict."""
    return KyberCiphertext(
        u=[np.array(arr, dtype=np.int64) for arr in data["u"]],
        v=np.array(data["v"], dtype=np.int64),
    )


# ──────────────────────────────────────────────────────────────────────
# PQC Transport
# ──────────────────────────────────────────────────────────────────────

class PQCTransportError(Exception):
    """Raised on decryption, integrity, or replay failures."""


class PQCTransport:
    """
    Post-Quantum encrypted transport for JSON payloads.

    Session-key strategy:
      - One Kyber KEM encapsulation per session (or per rotation interval)
      - Each message uses a unique random 12-byte AES-GCM nonce
      - Automatic key rotation every KEY_ROTATION_INTERVAL messages

    Replay protection:
      - Monotonic sequence counter per sender
      - Timestamp with MAX_CLOCK_SKEW window

    Envelope format (JSON with base64 fields):
      {
        "kem_ct":     base64(serialized KyberCiphertext),
        "nonce":      base64(12 bytes),
        "ciphertext": base64(AES-GCM ciphertext),
        "tag":        base64(AES-GCM tag),
        "seq":        int,
        "timestamp":  float,
        "integrity":  base64(SHA3-256 of plaintext),
        "key_epoch":  int
      }
    """

    def __init__(
        self,
        rotation_interval: int = KEY_ROTATION_INTERVAL,
        max_clock_skew: float = MAX_CLOCK_SKEW,
    ):
        self.kem = KyberKEM()
        self.rotation_interval = rotation_interval
        self.max_clock_skew = max_clock_skew

        # Keypair (set by keygen or by receiving a peer key)
        self._pk: Optional[KyberPublicKey] = None
        self._sk: Optional[KyberSecretKey] = None

        # Session encryption state (sender side)
        self._session_key: Optional[bytes] = None
        self._session_aes: Optional[AESGCM] = None
        self._session_kem_ct: Optional[KyberCiphertext] = None
        self._send_seq: int = 0
        self._key_epoch: int = 0

        # Peer public key (sender side — the aggregator's public key)
        self._peer_pk: Optional[KyberPublicKey] = None

        # Replay tracking (receiver side): {sender_id: last_seen_seq}
        self._last_seen_seq: Dict[str, int] = {}

        # Cached decapsulated keys by epoch: {key_epoch: session_key}
        self._decap_cache: Dict[str, bytes] = {}

    # ── Key Management ──────────────────────────────────────────────

    def keygen(self) -> KyberPublicKey:
        """Generate a Kyber keypair for this transport endpoint.

        Returns:
            The public key (to be sent to peers).
        """
        self._pk, self._sk = self.kem.keygen()
        return self._pk

    def get_public_key(self) -> Optional[KyberPublicKey]:
        """Return the local public key, if generated."""
        return self._pk

    def set_peer_public_key(self, pk: KyberPublicKey) -> None:
        """Store the peer's public key and establish initial session key.

        Called by the sensor after receiving the aggregator's public key.
        """
        self._peer_pk = pk
        self._establish_session_key()

    def _establish_session_key(self) -> None:
        """Perform KEM encapsulation to derive a new AES-256 session key."""
        if self._peer_pk is None:
            raise PQCTransportError("No peer public key set")

        ct, shared_secret = self.kem.encapsulate(self._peer_pk)
        self._session_key = shared_secret  # 32 bytes → AES-256
        self._session_aes = AESGCM(self._session_key)
        self._session_kem_ct = ct
        self._key_epoch += 1

    def rotate_key(self) -> None:
        """Force a session key rotation.

        Performs a new KEM encapsulation against the peer's public key,
        producing a fresh AES-256 key. The new KEM ciphertext will be
        included in the next encrypted message.
        """
        self._establish_session_key()

    # ── Encrypt (sender side) ───────────────────────────────────────

    def encrypt_payload(self, data: dict) -> bytes:
        """Serialize a dict to JSON, encrypt with AES-256-GCM, and return
        the PQC envelope as bytes.

        The KEM ciphertext is included so the receiver can decapsulate.
        A SHA3-256 integrity hash of the plaintext is embedded in the
        envelope for post-decryption verification.

        Args:
            data: JSON-serializable dict payload.

        Returns:
            UTF-8 encoded JSON envelope bytes.

        Raises:
            PQCTransportError: If no session key is established.
        """
        if self._session_aes is None or self._session_kem_ct is None:
            raise PQCTransportError(
                "No session key — call set_peer_public_key() first"
            )

        # Auto-rotate if interval exceeded
        if (
            self.rotation_interval > 0
            and self._send_seq > 0
            and self._send_seq % self.rotation_interval == 0
        ):
            self.rotate_key()

        # Serialize payload
        plaintext = json.dumps(data, separators=(",", ":"), default=str).encode("utf-8")

        # SHA3-256 integrity digest of plaintext
        integrity = hashlib.sha3_256(plaintext).digest()

        # Unique nonce
        nonce = os.urandom(NONCE_SIZE)

        # AES-256-GCM encrypt (ciphertext includes the 16-byte tag appended)
        ct_with_tag = self._session_aes.encrypt(nonce, plaintext, None)
        ciphertext = ct_with_tag[:-16]
        tag = ct_with_tag[-16:]

        # Increment sequence counter
        self._send_seq += 1

        # Build envelope
        envelope = {
            "kem_ct": json.dumps(serialize_ciphertext(self._session_kem_ct)),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "tag": base64.b64encode(tag).decode("ascii"),
            "seq": self._send_seq,
            "timestamp": time.time(),
            "integrity": base64.b64encode(integrity).decode("ascii"),
            "key_epoch": self._key_epoch,
        }

        return json.dumps(envelope, separators=(",", ":")).encode("utf-8")

    # ── Decrypt (receiver side) ─────────────────────────────────────

    def decrypt_payload(
        self,
        data: bytes,
        sender_id: str = "default",
    ) -> dict:
        """Decrypt a PQC envelope and return the original dict payload.

        Performs:
          1. Replay protection (seq + timestamp checks)
          2. KEM decapsulation (cached per key_epoch)
          3. AES-256-GCM decryption
          4. SHA3-256 integrity verification

        Args:
            data: UTF-8 encoded JSON envelope bytes.
            sender_id: Identifier for the sender (for replay tracking).

        Returns:
            The original dict payload.

        Raises:
            PQCTransportError: On replay, decryption, or integrity failure.
        """
        if self._sk is None:
            raise PQCTransportError(
                "No secret key — call keygen() first"
            )

        # Parse envelope
        try:
            envelope = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PQCTransportError(f"Malformed envelope: {exc}") from exc

        seq = envelope.get("seq", 0)
        timestamp = envelope.get("timestamp", 0.0)
        key_epoch = envelope.get("key_epoch", 0)

        # ── Replay protection ──
        # 1. Sequence check
        last_seq = self._last_seen_seq.get(sender_id, 0)
        if seq <= last_seq:
            raise PQCTransportError(
                f"Replay detected: seq={seq} <= last_seen={last_seq} "
                f"for sender '{sender_id}'"
            )

        # 2. Timestamp freshness check
        now = time.time()
        drift = abs(now - timestamp)
        if drift > self.max_clock_skew:
            raise PQCTransportError(
                f"Timestamp too stale: drift={drift:.1f}s > "
                f"max_clock_skew={self.max_clock_skew}s"
            )

        # ── Derive AES key via KEM decapsulation (cached per epoch) ──
        cache_key = f"{sender_id}:{key_epoch}"
        if cache_key in self._decap_cache:
            session_key = self._decap_cache[cache_key]
        else:
            kem_ct_data = json.loads(envelope["kem_ct"])
            kem_ct = deserialize_ciphertext(kem_ct_data)
            session_key = self.kem.decapsulate(self._sk, kem_ct)
            self._decap_cache[cache_key] = session_key

        aes = AESGCM(session_key)

        # ── Decode binary fields ──
        nonce = base64.b64decode(envelope["nonce"])
        ciphertext = base64.b64decode(envelope["ciphertext"])
        tag = base64.b64decode(envelope["tag"])
        expected_integrity = base64.b64decode(envelope["integrity"])

        # ── AES-256-GCM decrypt ──
        try:
            plaintext = aes.decrypt(nonce, ciphertext + tag, None)
        except Exception as exc:
            raise PQCTransportError(f"Decryption failed: {exc}") from exc

        # ── SHA3-256 integrity verification ──
        actual_integrity = hashlib.sha3_256(plaintext).digest()
        if actual_integrity != expected_integrity:
            raise PQCTransportError(
                "Integrity check failed: SHA3-256 mismatch after decryption"
            )

        # ── Accept: update replay state ──
        self._last_seen_seq[sender_id] = seq

        # Deserialize
        try:
            return json.loads(plaintext.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise PQCTransportError(
                f"Payload deserialization failed: {exc}"
            ) from exc

    # ── Diagnostics ─────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """Transport statistics."""
        return {
            "messages_encrypted": self._send_seq,
            "key_epoch": self._key_epoch,
            "rotation_interval": self.rotation_interval,
            "tracked_senders": len(self._last_seen_seq),
            "cached_keys": len(self._decap_cache),
        }
