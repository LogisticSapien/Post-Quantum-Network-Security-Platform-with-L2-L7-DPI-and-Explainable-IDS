"""
Dilithium-3 Digital Signature Module
=======================================
Post-quantum lattice-based digital signatures for distributed alert
integrity verification.

Implements a Dilithium-3-inspired signature scheme:
  • Module-LWE keygen with k×l matrix over Z_q[x]/(x^n+1)
  • Hash-then-sign with rejection sampling
  • Deterministic signing (hedged) using SHAKE-256

Parameters (FIPS 204 / Dilithium-3 inspired):
  q = 8380417 (prime, q-1 = 2^23 × ... supports NTT-256)
  n = 256     (polynomial degree)
  k = 6       (rows in public matrix A)
  l = 5       (columns / secret vector length)
  η = 2       (secret key coefficient bound)
  γ₁ = 2^19   (masking range)
  γ₂ = (q-1)/32 = 261888  (low-order rounding)
  τ = 49      (challenge weight)
  β = τ · η = 98  (norm bound after subtraction)

Integration:
  - SensorNode signs alert payloads before transmission
  - AggregationServer verifies signatures before processing
  - Key rotation every 24h via KEY_ROTATION message

Constraints: NumPy only for polynomial arithmetic.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Import NTT infrastructure from pqc.py
from pqc import _get_ntt_tables, _ntt_forward, _ntt_inverse


# ──────────────────────────────────────────────────────────────────────
# Dilithium Parameters
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DilithiumParams:
    """FIPS 204 Dilithium-3 parameters."""
    q: int = 8380417           # modulus
    n: int = 256               # polynomial degree
    k: int = 6                 # module rank (rows)
    l: int = 5                 # module rank (columns)
    eta: int = 2               # secret coefficient bound
    gamma1: int = 2**19        # masking range (524288)
    gamma2: int = (8380417 - 1) // 32  # = 261888, low-order rounding
    tau: int = 49              # challenge weight (# of ±1 entries)
    beta: int = 49 * 2         # = 98, norm bound: τ × η
    omega: int = 55            # max # of 1s in hint


DILITHIUM3 = DilithiumParams()


# ──────────────────────────────────────────────────────────────────────
# Polynomial Arithmetic (mod q, in Z_q[x]/(x^n+1))
# ──────────────────────────────────────────────────────────────────────

def _poly_add(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Add two polynomials mod q."""
    return (a.astype(np.int64) + b.astype(np.int64)) % q


def _poly_sub(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Subtract two polynomials mod q."""
    return (a.astype(np.int64) - b.astype(np.int64) + q) % q


def _poly_mul_ntt(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Multiply two polynomials in Z_q[x]/(x^n+1) using NTT."""
    a_ntt = _ntt_forward(a, q)
    b_ntt = _ntt_forward(b, q)
    c_ntt = (a_ntt.astype(np.int64) * b_ntt.astype(np.int64)) % q
    return _ntt_inverse(c_ntt, q)


def _center_reduce(a: np.ndarray, q: int) -> np.ndarray:
    """Center-reduce coefficients to range [-(q-1)/2, (q-1)/2]."""
    result = a.astype(np.int64) % q
    half_q = q // 2
    result[result > half_q] -= q
    return result


def _inf_norm(a: np.ndarray) -> int:
    """Infinity norm (max absolute coefficient)."""
    return int(np.max(np.abs(a)))


def _inf_norm_vec(vec: List[np.ndarray]) -> int:
    """Infinity norm of a vector of polynomials."""
    return max(_inf_norm(_center_reduce(p, DILITHIUM3.q)) for p in vec)


# ──────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────

def _sample_uniform_poly(seed: bytes, nonce: int, n: int, q: int) -> np.ndarray:
    """Sample a uniformly random polynomial in [0, q-1]."""
    # Use SHAKE-256 (via hashlib) to generate pseudorandom bytes
    h = hashlib.shake_256(seed + struct.pack('<H', nonce))
    coeffs = np.zeros(n, dtype=np.int64)
    idx = 0
    buf = b""
    offset = 0

    while idx < n:
        if offset + 3 > len(buf):
            buf = h.digest(n * 8)  # grab plenty of bytes
            offset = 0

        # Sample 3 bytes → 24 bits, acceptance bound
        b0 = buf[offset]
        b1 = buf[offset + 1]
        b2 = buf[offset + 2]
        offset += 3

        val = b0 | (b1 << 8) | ((b2 & 0x7F) << 16)  # 23 bits
        if val < q:
            coeffs[idx] = val
            idx += 1

    return coeffs


def _sample_cbd(seed: bytes, nonce: int, n: int, eta: int) -> np.ndarray:
    """Sample from centered binomial distribution CBD_η."""
    h = hashlib.shake_256(seed + struct.pack('<H', nonce))
    num_bytes = n * eta // 4
    buf = h.digest(num_bytes)

    coeffs = np.zeros(n, dtype=np.int64)
    bits = []
    for byte in buf:
        for bit in range(8):
            bits.append((byte >> bit) & 1)

    bit_idx = 0
    for i in range(n):
        a_sum = sum(bits[bit_idx + j] for j in range(eta))
        b_sum = sum(bits[bit_idx + eta + j] for j in range(eta))
        coeffs[i] = a_sum - b_sum
        bit_idx += 2 * eta

    return coeffs


def _sample_mask_poly(seed: bytes, nonce: int, n: int, gamma1: int, q: int) -> np.ndarray:
    """Sample a masking polynomial with coefficients in [-γ₁, γ₁]."""
    h = hashlib.shake_256(seed + struct.pack('<H', nonce))
    # 20-bit samples for γ₁ = 2^19
    num_bytes = n * 3  # 3 bytes per coefficient (24 bits, truncate)
    buf = h.digest(num_bytes)

    coeffs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        b0, b1, b2 = buf[3 * i], buf[3 * i + 1], buf[3 * i + 2]
        val = (b0 | (b1 << 8) | ((b2 & 0x0F) << 16))  # 20 bits → [0, 2^20 - 1]
        # Map to [-γ₁, γ₁]
        val = val % (2 * gamma1 + 1)
        coeffs[i] = (gamma1 - val) % q

    return coeffs


# ──────────────────────────────────────────────────────────────────────
# Rounding (Decompose / MakeHint / UseHint)
# ──────────────────────────────────────────────────────────────────────

def _decompose(r: int, alpha: int, q: int) -> Tuple[int, int]:
    """Decompose r into (r₁, r₀) where r ≡ r₁·α + r₀."""
    r = r % q
    r0 = r % alpha
    if r0 > alpha // 2:
        r0 -= alpha
    if r - r0 == q - 1:
        r1 = 0
        r0 -= 1
    else:
        r1 = (r - r0) // alpha
    return int(r1), int(r0)


def _high_bits(r: int, alpha: int, q: int) -> int:
    """Extract high-order bits."""
    r1, _ = _decompose(r, alpha, q)
    return r1


def _low_bits(r: int, alpha: int, q: int) -> int:
    """Extract low-order bits."""
    _, r0 = _decompose(r, alpha, q)
    return r0


def _high_bits_vec(vec: List[np.ndarray], alpha: int, q: int) -> List[np.ndarray]:
    """High bits for a vector of polynomials."""
    result = []
    for poly in vec:
        hi = np.array([_high_bits(int(c), alpha, q) for c in poly], dtype=np.int64)
        result.append(hi)
    return result


def _low_bits_vec(vec: List[np.ndarray], alpha: int, q: int) -> List[np.ndarray]:
    """Low bits for a vector of polynomials."""
    result = []
    for poly in vec:
        lo = np.array([_low_bits(int(c), alpha, q) for c in poly], dtype=np.int64)
        result.append(lo)
    return result


# ──────────────────────────────────────────────────────────────────────
# Challenge Polynomial
# ──────────────────────────────────────────────────────────────────────

def _sample_challenge(seed: bytes, tau: int, n: int) -> np.ndarray:
    """Sample challenge polynomial c with exactly τ nonzero ±1 entries."""
    h = hashlib.shake_256(seed)
    buf = h.digest(8 + n)  # plenty of bytes

    c = np.zeros(n, dtype=np.int64)

    # First 8 bytes → 64 sign bits
    signs = int.from_bytes(buf[:8], 'little')
    byte_idx = 8

    for i in range(n - tau, n):
        # Sample j ∈ [0, i] from the buffer
        j = buf[byte_idx] % (i + 1)
        byte_idx += 1

        c[i] = c[j]
        c[j] = 1 - 2 * (signs & 1)  # ±1
        signs >>= 1

    return c


# ──────────────────────────────────────────────────────────────────────
# Matrix-vector operations (module-lattice)
# ──────────────────────────────────────────────────────────────────────

def _expand_matrix(
    rho: bytes, k: int, l: int, n: int, q: int,
) -> List[List[np.ndarray]]:
    """Expand seed ρ into k×l public matrix A."""
    A = []
    for i in range(k):
        row = []
        for j in range(l):
            nonce = i * 256 + j
            row.append(_sample_uniform_poly(rho, nonce, n, q))
        A.append(row)
    return A


def _mat_vec_mul(
    A: List[List[np.ndarray]],
    s: List[np.ndarray],
    q: int,
) -> List[np.ndarray]:
    """Multiply k×l matrix A by l-vector s → k-vector."""
    k = len(A)
    l = len(s)
    n = len(s[0])
    result = [np.zeros(n, dtype=np.int64) for _ in range(k)]
    for i in range(k):
        for j in range(l):
            prod = _poly_mul_ntt(A[i][j], s[j], q)
            result[i] = _poly_add(result[i], prod, q)
    return result


def _vec_add(
    a: List[np.ndarray], b: List[np.ndarray], q: int,
) -> List[np.ndarray]:
    """Add two vectors of polynomials."""
    return [_poly_add(a[i], b[i], q) for i in range(len(a))]


def _vec_sub(
    a: List[np.ndarray], b: List[np.ndarray], q: int,
) -> List[np.ndarray]:
    """Subtract two vectors of polynomials."""
    return [_poly_sub(a[i], b[i], q) for i in range(len(a))]


def _scalar_poly_mul_vec(
    c: np.ndarray, v: List[np.ndarray], q: int,
) -> List[np.ndarray]:
    """Multiply each polynomial in vector v by scalar polynomial c."""
    return [_poly_mul_ntt(c, v[i], q) for i in range(len(v))]


# ──────────────────────────────────────────────────────────────────────
# Serialization helpers
# ──────────────────────────────────────────────────────────────────────

def _poly_to_bytes(poly: np.ndarray, q: int) -> bytes:
    """Serialize polynomial to bytes (3 bytes per coefficient)."""
    data = bytearray()
    for c in poly.astype(np.int64):
        val = int(c) % q
        data.extend(struct.pack('<I', val)[:3])
    return bytes(data)


def _poly_from_bytes(data: bytes, n: int, q: int) -> np.ndarray:
    """Deserialize polynomial from bytes."""
    coeffs = np.zeros(n, dtype=np.int64)
    for i in range(n):
        b = data[3 * i: 3 * i + 3]
        val = b[0] | (b[1] << 8) | (b[2] << 16)
        coeffs[i] = val % q
    return coeffs


def _vec_to_bytes(vec: List[np.ndarray], q: int) -> bytes:
    """Serialize vector of polynomials."""
    return b"".join(_poly_to_bytes(p, q) for p in vec)


def _vec_from_bytes(data: bytes, count: int, n: int, q: int) -> List[np.ndarray]:
    """Deserialize vector of polynomials."""
    poly_size = n * 3
    return [
        _poly_from_bytes(data[i * poly_size: (i + 1) * poly_size], n, q)
        for i in range(count)
    ]


# ──────────────────────────────────────────────────────────────────────
# Dilithium Signer
# ──────────────────────────────────────────────────────────────────────

class DilithiumSigner:
    """Dilithium-3 digital signature implementation.

    Provides keygen, sign, and verify operations for post-quantum
    digital signatures over alert payloads.

    Usage:
        signer = DilithiumSigner()
        pk, sk = signer.keygen()
        sig = signer.sign(message, sk)
        valid = signer.verify(message, sig, pk)
    """

    def __init__(self, params: DilithiumParams = DILITHIUM3):
        self.params = params
        # Warm up NTT cache for Dilithium's q
        _get_ntt_tables(params.n, params.q)

    def keygen(self) -> Tuple[bytes, bytes]:
        """Generate a Dilithium keypair.

        Returns:
            (public_key_bytes, secret_key_bytes) tuple.
        """
        p = self.params
        rho = os.urandom(32)        # seed for matrix A
        sigma = os.urandom(64)      # seed for secret vectors

        # Expand matrix A from ρ
        A = _expand_matrix(rho, p.k, p.l, p.n, p.q)

        # Sample secret vectors s₁ (l polys) and s₂ (k polys)
        s1 = [_sample_cbd(sigma, nonce, p.n, p.eta) for nonce in range(p.l)]
        s2 = [_sample_cbd(sigma, p.l + nonce, p.n, p.eta) for nonce in range(p.k)]

        # Compute t = A·s₁ + s₂
        t = _vec_add(_mat_vec_mul(A, s1, p.q), s2, p.q)

        # Serialize
        pk = rho + _vec_to_bytes(t, p.q)
        sk = rho + sigma + _vec_to_bytes(s1, p.q) + _vec_to_bytes(s2, p.q) + _vec_to_bytes(t, p.q)

        return pk, sk

    def sign(self, message: bytes, sk: bytes) -> bytes:
        """Sign a message with the secret key.

        Uses deterministic signing with rejection sampling.

        Args:
            message: Message bytes to sign.
            sk: Secret key bytes from keygen().

        Returns:
            Signature bytes.
        """
        p = self.params
        poly_bytes = p.n * 3

        # Parse secret key
        rho = sk[:32]
        sigma = sk[32:96]
        offset = 96
        s1 = _vec_from_bytes(sk[offset:offset + p.l * poly_bytes], p.l, p.n, p.q)
        offset += p.l * poly_bytes
        s2 = _vec_from_bytes(sk[offset:offset + p.k * poly_bytes], p.k, p.n, p.q)
        offset += p.k * poly_bytes
        t = _vec_from_bytes(sk[offset:offset + p.k * poly_bytes], p.k, p.n, p.q)

        # Expand A
        A = _expand_matrix(rho, p.k, p.l, p.n, p.q)

        # Compute µ = H(ρ || H(pk) || message)
        pk_hash = hashlib.shake_256(rho + _vec_to_bytes(t, p.q)).digest(64)
        mu = hashlib.shake_256(pk_hash + message).digest(64)

        # Rejection sampling loop
        kappa = 0
        max_attempts = 500

        while kappa < max_attempts:
            # Sample masking vector y
            seed_y = sigma + mu + struct.pack('<H', kappa)
            rho_prime = hashlib.shake_256(seed_y).digest(64)

            y = [
                _sample_mask_poly(rho_prime, nonce, p.n, p.gamma1, p.q)
                for nonce in range(p.l)
            ]

            # w = A·y
            w = _mat_vec_mul(A, y, p.q)

            # HighBits(w)
            w1 = _high_bits_vec(w, 2 * p.gamma2, p.q)

            # Hash w1 for challenge
            w1_bytes = b""
            for poly in w1:
                w1_bytes += bytes([int(c) & 0xFF for c in poly])

            c_seed = hashlib.shake_256(mu + w1_bytes).digest(32)
            c_poly = _sample_challenge(c_seed, p.tau, p.n)

            # z = y + c·s₁
            cs1 = _scalar_poly_mul_vec(c_poly, s1, p.q)
            z = _vec_add(y, cs1, p.q)

            # Center-reduce z and check norm bound
            z_centered = [_center_reduce(zi, p.q) for zi in z]
            z_norm = max(_inf_norm(zi) for zi in z_centered)

            if z_norm >= p.gamma1 - p.beta:
                kappa += 1
                continue

            # Check low bits of w - c·s₂
            cs2 = _scalar_poly_mul_vec(c_poly, s2, p.q)
            r = _vec_sub(w, cs2, p.q)
            r0 = _low_bits_vec(r, 2 * p.gamma2, p.q)
            r0_norm = max(_inf_norm(r0i) for r0i in r0)

            if r0_norm >= p.gamma2 - p.beta:
                kappa += 1
                continue

            # Signature found! Pack (c_seed, z)
            sig = c_seed + _vec_to_bytes(z, p.q)
            return sig

        raise RuntimeError(
            f"Dilithium signing failed after {max_attempts} rejection sampling rounds"
        )

    def verify(self, message: bytes, signature: bytes, pk: bytes) -> bool:
        """Verify a signature against a message and public key.

        Args:
            message: Original message bytes.
            signature: Signature bytes from sign().
            pk: Public key bytes from keygen().

        Returns:
            True if signature is valid, False otherwise.
        """
        p = self.params
        poly_bytes = p.n * 3

        try:
            # Parse public key
            rho = pk[:32]
            t = _vec_from_bytes(pk[32:32 + p.k * poly_bytes], p.k, p.n, p.q)

            # Parse signature
            c_seed = signature[:32]
            z = _vec_from_bytes(signature[32:32 + p.l * poly_bytes], p.l, p.n, p.q)

            # Check z norm bound
            z_centered = [_center_reduce(zi, p.q) for zi in z]
            z_norm = max(_inf_norm(zi) for zi in z_centered)
            if z_norm >= p.gamma1 - p.beta:
                return False

            # Reconstruct challenge
            c_poly = _sample_challenge(c_seed, p.tau, p.n)

            # Expand A
            A = _expand_matrix(rho, p.k, p.l, p.n, p.q)

            # Compute µ
            pk_hash = hashlib.shake_256(pk).digest(64)
            mu = hashlib.shake_256(pk_hash + message).digest(64)

            # w' = A·z − c·t
            Az = _mat_vec_mul(A, z, p.q)
            ct = _scalar_poly_mul_vec(c_poly, t, p.q)
            w_prime = _vec_sub(Az, ct, p.q)

            # HighBits(w')
            w1_prime = _high_bits_vec(w_prime, 2 * p.gamma2, p.q)

            # Recompute challenge hash
            w1_bytes = b""
            for poly in w1_prime:
                w1_bytes += bytes([int(c) & 0xFF for c in poly])

            c_seed_prime = hashlib.shake_256(mu + w1_bytes).digest(32)

            return c_seed == c_seed_prime

        except Exception:
            return False

    # ── Key Serialization ──

    @staticmethod
    def serialize_pk(pk: bytes) -> bytes:
        """Serialize public key (already bytes, identity op)."""
        return pk

    @staticmethod
    def deserialize_pk(data: bytes) -> bytes:
        """Deserialize public key."""
        return data

    @staticmethod
    def pk_hash(pk: bytes) -> str:
        """Compute a short hash of the public key for identification."""
        return hashlib.sha3_256(pk).hexdigest()[:16]

    @staticmethod
    def save_encrypted_sk(
        sk: bytes,
        encryption_key: bytes,
        path: str,
    ) -> None:
        """Save secret key encrypted with AES-256-GCM.

        Args:
            sk: Secret key bytes.
            encryption_key: 32-byte AES key.
            path: File path to save to.
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(12)
        aes = AESGCM(encryption_key)
        ciphertext = aes.encrypt(nonce, sk, b"dilithium-sk-v1")
        with open(path, 'wb') as f:
            f.write(nonce + ciphertext)

    @staticmethod
    def load_encrypted_sk(
        encryption_key: bytes,
        path: str,
    ) -> bytes:
        """Load and decrypt a secret key.

        Args:
            encryption_key: 32-byte AES key.
            path: File path to load from.

        Returns:
            Secret key bytes.
        """
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        with open(path, 'rb') as f:
            data = f.read()
        nonce = data[:12]
        ciphertext = data[12:]
        aes = AESGCM(encryption_key)
        return aes.decrypt(nonce, ciphertext, b"dilithium-sk-v1")


# ──────────────────────────────────────────────────────────────────────
# Alert Signing Helpers (for distributed.py integration)
# ──────────────────────────────────────────────────────────────────────

def sign_alert_payload(payload: bytes, sk: bytes) -> bytes:
    """Sign an alert payload. Returns signature bytes."""
    signer = DilithiumSigner()
    return signer.sign(payload, sk)


def verify_alert_signature(
    payload: bytes, signature: bytes, pk: bytes,
) -> bool:
    """Verify an alert payload signature."""
    signer = DilithiumSigner()
    return signer.verify(payload, signature, pk)
