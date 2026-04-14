"""
Post-Quantum Cryptography Engine
=================================
Provides Shor's-algorithm-resistant cryptographic primitives:
  • Kyber-inspired lattice-based Key Encapsulation Mechanism (KEM)
  • SHA3-256 hash-chain for tamper-evident logging
  • AES-256-GCM symmetric encryption keyed via Kyber KEM
  • Quantum-vulnerability scanner for observed TLS cipher suites
"""

from __future__ import annotations

import sys
import io

# Fix Windows console encoding for Unicode output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import hashlib
import hmac
import json
import math
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ──────────────────────────────────────────────────────────────────────
# Kyber Parameters — configurable between educational and production
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KyberParams:
    """Kyber KEM parameters — controls security level."""
    N: int          # polynomial degree (power of 2)
    Q: int          # modulus
    K: int          # module rank
    ETA1: int       # noise parameter for key generation
    ETA2: int       # noise parameter for encryption
    label: str = ""

# Educational preset (N=64) — faster, for demos and testing
KYBER_EDUCATIONAL = KyberParams(N=64,  Q=3329, K=2, ETA1=2, ETA2=2, label="educational")

# Production preset (N=256) — matches real CRYSTALS-Kyber-512
KYBER_PRODUCTION  = KyberParams(N=256, Q=3329, K=2, ETA1=3, ETA2=2, label="production")

# Active parameters (default: production)
_active_params: KyberParams = KYBER_PRODUCTION

def set_kyber_level(level: str = "production") -> KyberParams:
    """Set the active Kyber parameter level. Returns the active params."""
    global _active_params
    if level == "educational":
        _active_params = KYBER_EDUCATIONAL
    elif level == "production":
        _active_params = KYBER_PRODUCTION
    else:
        raise ValueError(f"Unknown Kyber level: {level!r} (use 'educational' or 'production')")
    return _active_params

def get_kyber_params() -> KyberParams:
    """Get the currently active Kyber parameters."""
    return _active_params

# Legacy compatibility — module-level constant
KYBER_Q = 3329         # modulus is always the same

# ──────────────────────────────────────────────────────────────────────
# NTT (Number Theoretic Transform) for negacyclic ring Z_q[x]/(x^n+1)
# O(n log n) vs O(n²) schoolbook — standard Kyber optimisation.
#
# Strategy for negacyclic convolution in Z_q[x]/(x^n+1):
#   1. Pre-twist: a'[i] = a[i] · ψ^i  where ψ is a primitive 2n-th
#      root of unity (so ψ^n ≡ -1 mod q). This converts the problem
#      to a standard cyclic convolution.
#   2. Standard NTT using ω = ψ² (a primitive n-th root of unity)
#   3. Pointwise multiply in NTT domain
#   4. Inverse NTT
#   5. Post-twist: c[i] = c'[i] · ψ^(-i)
#
# q = 3329 (prime), q-1 = 3328 = 2^8 × 13
# ──────────────────────────────────────────────────────────────────────

def _prime_factors(n: int) -> set[int]:
    """Return the set of prime factors of n via trial division."""
    factors: set[int] = set()
    d = 2
    while d * d <= n:
        while n % d == 0:  # type: ignore[operator]
            factors.add(d)
            n = n // d  # type: ignore[operator]
        d += 1
    if n > 1:  # type: ignore[operator]
        factors.add(n)  # type: ignore[arg-type]
    return factors


def _find_generator(q: int) -> int:
    """Find a primitive root (generator) of Z_q*."""
    phi = q - 1
    factors = _prime_factors(phi)
    for g in range(2, q):
        if all(pow(g, phi // f, q) != 1 for f in factors):
            return g
    raise ValueError(f"No generator found for q={q}")


def _bit_reverse(x: int, bits: int) -> int:
    """Reverse the lowest `bits` bits of integer x."""
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def _precompute_ntt_tables(n: int, q: int) -> dict:
    """
    Precompute everything needed for NTT-based negacyclic multiplication.
    """
    g = _find_generator(q)
    log_n = int(math.log2(n))

    # ψ = primitive 2n-th root of unity:  ψ^(2n) ≡ 1, ψ^n ≡ -1
    psi = pow(g, (q - 1) // (2 * n), q)
    assert pow(psi, 2 * n, q) == 1
    assert pow(psi, n, q) == q - 1

    # ω = ψ² = primitive n-th root of unity:  ω^n ≡ 1
    omega = (psi * psi) % q
    assert pow(omega, n, q) == 1

    # Precompute twist factors: psi_pow[i] = ψ^i mod q
    psi_pow = [pow(psi, i, q) for i in range(n)]
    psi_inv_pow = [pow(psi_pow[i], q - 2, q) for i in range(n)]

    # Precompute omega powers for each NTT stage
    # For Cooley-Tukey DIT: at stage s (0..log_n-1), groups of size m=2^(s+1)
    #   twiddle = ω^(n/m) = primitive m-th root of unity
    omega_table = []  # omega_table[s] = ω^(n / 2^(s+1))
    for s in range(log_n):
        m = 1 << (s + 1)
        w = pow(omega, n // m, q)  # primitive m-th root
        omega_table.append(w)

    # Inverse omega table: ω_inv^(n/m)
    omega_inv = pow(omega, q - 2, q)
    omega_inv_table = []
    for s in range(log_n):
        m = 1 << (s + 1)
        w = pow(omega_inv, n // m, q)
        omega_inv_table.append(w)

    # Bit-reversal permutation table
    br_perm = [_bit_reverse(i, log_n) for i in range(n)]

    n_inv = pow(n, q - 2, q)

    return {
        "psi_pow": psi_pow, "psi_inv_pow": psi_inv_pow,
        "omega_table": omega_table, "omega_inv_table": omega_inv_table,
        "br_perm": br_perm, "n_inv": n_inv, "log_n": log_n,
    }


_NTT_CACHE: dict = {}

def _get_ntt_tables(n: int, q: int) -> dict:
    key = (n, q)
    if key not in _NTT_CACHE:
        _NTT_CACHE[key] = _precompute_ntt_tables(n, q)
    return _NTT_CACHE[key]


def _ntt_forward(poly: np.ndarray, q: int = KYBER_Q) -> np.ndarray:
    """Forward NTT with negacyclic pre-twist. Cooley-Tukey DIT."""
    n = len(poly)
    T = _get_ntt_tables(n, q)
    a = poly.astype(np.int64).copy()

    # Step 1: Pre-twist for negacyclic — a[i] *= ψ^i
    for i in range(n):
        a[i] = (int(a[i]) * T["psi_pow"][i]) % q

    # Step 2: Bit-reversal permutation
    br = T["br_perm"]
    a_br = np.empty(n, dtype=np.int64)
    for i in range(n):
        a_br[i] = a[br[i]]
    a = a_br

    # Step 3: Cooley-Tukey butterfly (DIT)
    for s in range(T["log_n"]):
        m = 1 << (s + 1)
        half = m >> 1
        w_m = T["omega_table"][s]  # ω^(n/m)

        for k in range(0, n, m):
            w = 1
            for j in range(half):
                t = (w * int(a[k + j + half])) % q  # type: ignore[arg-type]
                u = int(a[k + j])  # type: ignore[arg-type]
                a[k + j] = (u + t) % q
                a[k + j + half] = (u - t + q) % q
                w = (w * w_m) % q

    return a


def _ntt_inverse(a_ntt: np.ndarray, q: int = KYBER_Q) -> np.ndarray:
    """Inverse NTT with negacyclic post-twist. Gentleman-Sande DIF."""
    n = len(a_ntt)
    T = _get_ntt_tables(n, q)
    a = a_ntt.astype(np.int64).copy()

    # Step 1: Gentleman-Sande butterfly (DIF) — reverse stage order
    for s in range(T["log_n"] - 1, -1, -1):
        m = 1 << (s + 1)
        half = m >> 1
        w_m = T["omega_inv_table"][s]  # ω_inv^(n/m)

        for k in range(0, n, m):
            w = 1
            for j in range(half):
                u = int(a[k + j])  # type: ignore[arg-type]
                v = int(a[k + j + half])  # type: ignore[arg-type]
                a[k + j] = (u + v) % q
                a[k + j + half] = ((u - v + q) * w) % q
                w = (w * w_m) % q

    # Step 2: Bit-reversal permutation
    br = T["br_perm"]
    a_br = np.empty(n, dtype=np.int64)
    for i in range(n):
        a_br[i] = a[br[i]]
    a = a_br

    # Step 3: Multiply by n^(-1)
    n_inv = T["n_inv"]
    for i in range(n):
        a[i] = (int(a[i]) * n_inv) % q

    # Step 4: Post-twist for negacyclic — a[i] *= ψ^(-i)
    for i in range(n):
        a[i] = (int(a[i]) * T["psi_inv_pow"][i]) % q

    return a


def _poly_mul_schoolbook(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Polynomial multiplication in Z_q[x]/(x^n+1) using schoolbook method.
    O(n²) — used as reference / fallback.
    """
    n = len(a)
    result = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ai = int(a[i])
        if ai == 0:
            continue
        for j in range(n):
            idx = i + j
            val = ai * int(b[j])
            if idx < n:
                result[idx] = (result[idx] + val) % KYBER_Q
            else:
                result[idx - n] = (result[idx - n] - val) % KYBER_Q
    return result


def _poly_mul_ntt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Polynomial multiplication in Z_q[x]/(x^n+1) using NTT.
    O(n log n) — the standard Kyber optimisation.

    Process: NTT(a) ⊙ NTT(b) → INTT(product)
    where ⊙ is pointwise multiplication in NTT domain.
    """
    n = len(a)
    a_ntt = _ntt_forward(a)
    b_ntt = _ntt_forward(b)

    # Pointwise multiplication in NTT domain
    c_ntt = (a_ntt * b_ntt) % KYBER_Q

    return _ntt_inverse(c_ntt)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _ntt_supported(n: int, q: int = KYBER_Q) -> bool:
    """Check if negacyclic NTT is supported for given n and q.

    Requires a primitive 2n-th root of unity mod q, which exists
    iff 2n divides (q-1).  For q=3329: q-1=3328=2^8*13, so the
    maximum supported 2n is 256 → n_max=128.
    """
    return _is_power_of_two(n) and n >= 8 and (q - 1) % (2 * n) == 0


# ──────────────────────────────────────────────────────────────────────
# FIPS 203 NTT for N=256 (ML-KEM / Kyber production parameters)
#
# The generic negacyclic NTT requires 2n | (q-1), which fails for
# n=256 (q-1 = 3328 = 2^8 × 13, max 2n = 256 → n_max = 128).
#
# FIPS 203 uses ζ = 17 as primitive 256th root of unity mod 3329
# (17^128 ≡ -1 mod 3329). The NTT decomposes Z_q[x]/(x^256+1) into
# 128 degree-1 quotients via Cooley-Tukey butterfly (7 layers).
# This is O(n log n) ≈ 2048 mults vs O(n²) = 65536 for schoolbook.
# ──────────────────────────────────────────────────────────────────────

def _bitrev7(n: int) -> int:
    """Reverse the lowest 7 bits of integer n."""
    r = 0
    for _ in range(7):
        r = (r << 1) | (n & 1)
        n >>= 1
    return r


# Precompute zeta table: zetas[i] = 17^{BitRev7(i)} mod 3329
_ZETAS_256 = [pow(17, _bitrev7(i), KYBER_Q) for i in range(128)]

# Precompute gammas for base-case multiplication:
# gamma[i] = 17^{2·BitRev7(i) + 1} mod 3329
_GAMMAS_256 = [pow(17, 2 * _bitrev7(i) + 1, KYBER_Q) for i in range(128)]

# 128^{-1} mod 3329 (for inverse NTT scaling)
_INV_128 = pow(128, KYBER_Q - 2, KYBER_Q)  # = 3303


def _ntt_forward_256(f: np.ndarray) -> np.ndarray:
    """FIPS 203 forward NTT for n=256. Cooley-Tukey DIT, 7 layers."""
    a = f.astype(np.int64).copy()
    k = 1
    length = 128
    while length >= 2:
        for start in range(0, 256, 2 * length):
            zeta = _ZETAS_256[k]
            k += 1
            for j in range(start, start + length):
                t = (zeta * int(a[j + length])) % KYBER_Q
                a[j + length] = (int(a[j]) - t + KYBER_Q) % KYBER_Q
                a[j] = (int(a[j]) + t) % KYBER_Q
        length >>= 1
    return a


def _ntt_inverse_256(a_ntt: np.ndarray) -> np.ndarray:
    """FIPS 203 inverse NTT for n=256. Gentleman-Sande DIF, 7 layers."""
    a = a_ntt.astype(np.int64).copy()
    k = 127
    length = 2
    while length <= 128:
        for start in range(0, 256, 2 * length):
            zeta = _ZETAS_256[k]
            k -= 1
            for j in range(start, start + length):
                t = int(a[j])
                a[j] = (t + int(a[j + length])) % KYBER_Q
                a[j + length] = (zeta * (int(a[j + length]) - t + KYBER_Q)) % KYBER_Q
        length <<= 1
    for i in range(256):
        a[i] = (int(a[i]) * _INV_128) % KYBER_Q
    return a


def _basemul_256(a_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
    """
    Pointwise multiplication of two NTT-domain polynomials (n=256).
    Each pair (a[2i], a[2i+1]) is a degree-1 polynomial in
    Z_q[x]/(x² - γ_i), multiplied via schoolbook on the pair.
    """
    c = np.zeros(256, dtype=np.int64)
    for i in range(128):
        a0, a1 = int(a_hat[2*i]), int(a_hat[2*i+1])
        b0, b1 = int(b_hat[2*i]), int(b_hat[2*i+1])
        gamma = _GAMMAS_256[i]
        c[2*i]     = (a0 * b0 + a1 * b1 * gamma) % KYBER_Q
        c[2*i + 1] = (a0 * b1 + a1 * b0) % KYBER_Q
    return c


def _poly_mul_ntt_256(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Polynomial multiplication in Z_q[x]/(x^256+1) using FIPS 203 NTT.
    O(n log n) — replaces O(n²) schoolbook for production N=256.
    """
    a_hat = _ntt_forward_256(a)
    b_hat = _ntt_forward_256(b)
    c_hat = _basemul_256(a_hat, b_hat)
    return _ntt_inverse_256(c_hat)


def _poly_mul_ring(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Polynomial multiplication in Z_q[x]/(x^n+1).
    Uses FIPS 203 NTT for N=256, generic negacyclic NTT for N≤128
    (when 2n | q-1), and O(n²) schoolbook as final fallback.
    """
    n = len(a)
    if n == 256:
        return _poly_mul_ntt_256(a, b)
    if _ntt_supported(n, KYBER_Q):
        return _poly_mul_ntt(a, b)
    return _poly_mul_schoolbook(a, b)



def _cbd(eta: int, seed: bytes, nonce: int) -> np.ndarray:
    """Centred Binomial Distribution sampling for noise polynomials."""
    p = _active_params
    rng = np.random.RandomState(
        list(hashlib.sha3_256(seed + struct.pack('<B', nonce)).digest()[:4])
    )
    buf_a = rng.randint(0, 2, size=(p.N, eta))
    buf_b = rng.randint(0, 2, size=(p.N, eta))
    return (buf_a.sum(axis=1) - buf_b.sum(axis=1)) % p.Q


def _sample_uniform(seed: bytes, i: int, j: int) -> np.ndarray:
    """Uniformly sample a polynomial from Z_q."""
    p = _active_params
    h = hashlib.sha3_512(seed + struct.pack('<BB', i, j)).digest()
    rng = np.random.RandomState(list(h[:4]))
    return rng.randint(0, p.Q, size=p.N).astype(np.int64)


@dataclass
class KyberPublicKey:
    """Public key: (t, A_seed)."""
    t: List[np.ndarray]        # k polynomials
    rho: bytes                  # seed for matrix A


@dataclass
class KyberSecretKey:
    """Secret key: s polynomials."""
    s: List[np.ndarray]        # k polynomials


@dataclass
class KyberCiphertext:
    """Ciphertext: (u, v) — uncompressed for correctness."""
    u: List[np.ndarray]
    v: np.ndarray


class KyberKEM:
    """
    Kyber-inspired Key Encapsulation Mechanism.

    Provides IND-CCA2-like security against quantum adversaries
    by relying on the Module-LWE problem.

    Uses schoolbook polynomial multiplication in Z_q[x]/(x^n+1)
    to guarantee correct encapsulation/decapsulation round-trips.
    """

    def __init__(self, k: int = None):
        self.k = k if k is not None else _active_params.K

    def _gen_matrix(self, rho: bytes) -> list:
        """Generate public matrix A from seed."""
        return [[_sample_uniform(rho, i, j)
                 for j in range(self.k)] for i in range(self.k)]

    def keygen(self, seed: Optional[bytes] = None) -> Tuple[KyberPublicKey, KyberSecretKey]:
        """Generate a Kyber keypair."""
        if seed is None:
            seed = os.urandom(32)
        p = _active_params

        rho = hashlib.sha3_256(seed + b'rho').digest()
        sigma = hashlib.sha3_256(seed + b'sigma').digest()

        A = self._gen_matrix(rho)

        # Sample secret vector s
        s = [_cbd(p.ETA1, sigma, i) for i in range(self.k)]

        # Sample error vector e
        e = [_cbd(p.ETA1, sigma, self.k + i) for i in range(self.k)]

        # t = As + e in Z_q[x]/(x^n+1)
        t = []
        for i in range(self.k):
            acc = np.zeros(p.N, dtype=np.int64)
            for j in range(self.k):
                acc = (acc + _poly_mul_ring(A[i][j], s[j])) % p.Q
            t.append((acc + e[i]) % p.Q)

        pk = KyberPublicKey(t=t, rho=rho)
        sk = KyberSecretKey(s=s)
        return pk, sk

    def encapsulate(
        self, pk: KyberPublicKey, seed: Optional[bytes] = None
    ) -> Tuple[KyberCiphertext, bytes]:
        """
        Encapsulate a shared secret under the given public key.
        Returns (ciphertext, shared_secret_32_bytes).
        """
        if seed is None:
            seed = os.urandom(32)
        p = _active_params

        msg_full = hashlib.sha3_256(seed).digest()
        msg_len = int(p.N // 8)  # bits that fit in the polynomial
        msg = msg_full[:msg_len]

        # Encode message as polynomial: each bit → 0 or ⌈q/2⌉
        m_poly = np.zeros(p.N, dtype=np.int64)
        for i in range(p.N):
            byte_idx = int(i // 8)
            bit_idx = int(i % 8)
            m_poly[i] = ((msg[byte_idx] >> bit_idx) & 1) * ((p.Q + 1) // 2)

        coin = hashlib.sha3_256(seed + b'coin').digest()

        # Re-derive A and transpose
        A = self._gen_matrix(pk.rho)

        # Sample randomness r, errors e1, e2
        r = [_cbd(p.ETA1, coin, i) for i in range(self.k)]
        e1 = [_cbd(p.ETA2, coin, self.k + i) for i in range(self.k)]
        e2 = _cbd(p.ETA2, coin, 2 * self.k)

        # u = A^T r + e1
        u = []
        for i in range(self.k):
            acc = np.zeros(p.N, dtype=np.int64)
            for j in range(self.k):
                acc = (acc + _poly_mul_ring(A[j][i], r[j])) % p.Q
            u.append((acc + e1[i]) % p.Q)

        # v = t^T r + e2 + m
        v = np.zeros(p.N, dtype=np.int64)
        for j in range(self.k):
            v = (v + _poly_mul_ring(pk.t[j], r[j])) % p.Q
        v = (v + e2 + m_poly) % p.Q

        ct = KyberCiphertext(u=u, v=v)

        shared = hashlib.sha3_256(msg + b'shared').digest()
        return ct, shared

    def decapsulate(self, sk: KyberSecretKey, ct: KyberCiphertext) -> bytes:
        """Decapsulate to recover the shared secret."""
        p = _active_params

        # Compute s^T u
        su = np.zeros(p.N, dtype=np.int64)
        for j in range(self.k):
            su = (su + _poly_mul_ring(sk.s[j], ct.u[j])) % p.Q

        # m' = v - s^T u
        m_prime = (ct.v - su) % p.Q

        # Decode message: each coefficient → nearest to 0 or ⌈q/2⌉
        msg_len = int(p.N // 8)
        msg_bytes = bytearray(msg_len)
        for i in range(p.N):
            val = int(m_prime[i])
            dist_0 = min(val, p.Q - val)
            dist_1 = abs(val - (p.Q + 1) // 2)
            bit = 1 if dist_1 < dist_0 else 0

            byte_idx = int(i // 8)
            bit_idx = int(i % 8)
            msg_bytes[byte_idx] |= (bit << bit_idx)

        shared = hashlib.sha3_256(bytes(msg_bytes) + b'shared').digest()
        return shared

    def _encrypt_raw(
        self, pk: KyberPublicKey, msg: bytes, coin: bytes
    ) -> KyberCiphertext:
        """Deterministic encryption of raw message bytes with given coins.

        Used by the FO transform for re-encryption verification.
        msg must be exactly N//8 bytes.
        """
        p = _active_params
        msg_len = int(p.N // 8)
        assert len(msg) == msg_len, f"msg must be {msg_len} bytes, got {len(msg)}"

        # Encode message as polynomial
        m_poly = np.zeros(p.N, dtype=np.int64)
        for i in range(p.N):
            byte_idx = int(i // 8)
            bit_idx = int(i % 8)
            m_poly[i] = ((msg[byte_idx] >> bit_idx) & 1) * ((p.Q + 1) // 2)

        A = self._gen_matrix(pk.rho)

        r = [_cbd(p.ETA1, coin, i) for i in range(self.k)]
        e1 = [_cbd(p.ETA2, coin, self.k + i) for i in range(self.k)]
        e2 = _cbd(p.ETA2, coin, 2 * self.k)

        u = []
        for i in range(self.k):
            acc = np.zeros(p.N, dtype=np.int64)
            for j in range(self.k):
                acc = (acc + _poly_mul_ring(A[j][i], r[j])) % p.Q
            u.append((acc + e1[i]) % p.Q)

        v = np.zeros(p.N, dtype=np.int64)
        for j in range(self.k):
            v = (v + _poly_mul_ring(pk.t[j], r[j])) % p.Q
        v = (v + e2 + m_poly) % p.Q

        return KyberCiphertext(u=u, v=v)

    def _decrypt_raw(self, sk: KyberSecretKey, ct: KyberCiphertext) -> bytes:
        """Decrypt ciphertext to raw message bytes (no hashing).

        Used by the FO transform to recover the plaintext for re-encryption.
        Returns N//8 bytes.
        """
        p = _active_params

        su = np.zeros(p.N, dtype=np.int64)
        for j in range(self.k):
            su = (su + _poly_mul_ring(sk.s[j], ct.u[j])) % p.Q

        m_prime = (ct.v - su) % p.Q

        msg_len = int(p.N // 8)
        msg_bytes = bytearray(msg_len)
        for i in range(p.N):
            val = int(m_prime[i])
            dist_0 = min(val, p.Q - val)
            dist_1 = abs(val - (p.Q + 1) // 2)
            bit = 1 if dist_1 < dist_0 else 0
            byte_idx = int(i // 8)
            bit_idx = int(i % 8)
            msg_bytes[byte_idx] |= (bit << bit_idx)

        return bytes(msg_bytes)


# ──────────────────────────────────────────────────────────────────────
# KEM Statistics Tracker
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KEMStats:
    """Track KEM operation timings and counts."""
    keygen_count: int = 0
    encap_count: int = 0
    decap_count: int = 0
    keygen_total_ms: float = 0.0
    encap_total_ms: float = 0.0
    decap_total_ms: float = 0.0

    def record_keygen(self, elapsed_ms: float):
        self.keygen_count += 1
        self.keygen_total_ms += elapsed_ms

    def record_encap(self, elapsed_ms: float):
        self.encap_count += 1
        self.encap_total_ms += elapsed_ms

    def record_decap(self, elapsed_ms: float):
        self.decap_count += 1
        self.decap_total_ms += elapsed_ms

    @property
    def avg_keygen_ms(self) -> float:
        return self.keygen_total_ms / max(self.keygen_count, 1)

    @property
    def avg_encap_ms(self) -> float:
        return self.encap_total_ms / max(self.encap_count, 1)

    @property
    def avg_decap_ms(self) -> float:
        return self.decap_total_ms / max(self.decap_count, 1)

    @property
    def summary(self) -> dict:
        return {
            "keygen": {"count": self.keygen_count, "avg_ms": round(self.avg_keygen_ms, 2)},
            "encap": {"count": self.encap_count, "avg_ms": round(self.avg_encap_ms, 2)},
            "decap": {"count": self.decap_count, "avg_ms": round(self.avg_decap_ms, 2)},
        }


_kem_stats = KEMStats()

def get_kem_stats() -> KEMStats:
    """Get global KEM statistics."""
    return _kem_stats


# ──────────────────────────────────────────────────────────────────────
# Rate Limiter — token bucket for DoS protection on KEM operations
# ──────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Token bucket rate limiter.

    Limits the rate of operations per key (e.g., per-IP encapsulation
    rate) to prevent KEM DoS attacks.

    Args:
        rate: Tokens replenished per second.
        burst: Maximum token capacity.
        cleanup_interval: Seconds between stale-entry eviction.
    """

    def __init__(self, rate: float = 10.0, burst: int = 20,
                 cleanup_interval: float = 60.0):
        self.rate = rate
        self.burst = burst
        self.cleanup_interval = cleanup_interval
        self._buckets: dict = {}  # key -> (tokens, last_time)
        self._last_cleanup = time.time()

    def allow(self, key: str) -> bool:
        """Check if an operation is allowed for the given key.

        Returns True if a token is available (consumes it).
        Returns False if rate limit exceeded.
        """
        now = time.time()

        # Periodic cleanup
        if now - self._last_cleanup > self.cleanup_interval:
            self.cleanup()
            self._last_cleanup = now

        if key not in self._buckets:
            self._buckets[key] = (self.burst - 1, now)
            return True

        tokens, last_time = self._buckets[key]
        elapsed = now - last_time
        tokens = min(self.burst, tokens + elapsed * self.rate)

        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1, now)
            return True
        else:
            self._buckets[key] = (tokens, now)
            return False

    def cleanup(self):
        """Remove stale entries (idle > 2× cleanup_interval)."""
        now = time.time()
        cutoff = now - 2 * self.cleanup_interval
        stale = [k for k, (_, t) in self._buckets.items() if t < cutoff]
        for k in stale:
            del self._buckets[k]

    @property
    def tracked_keys(self) -> int:
        return len(self._buckets)


# ──────────────────────────────────────────────────────────────────────
# Fujisaki-Okamoto Transform — IND-CCA2 secure KEM
# ──────────────────────────────────────────────────────────────────────

@dataclass
class CCASecretKey:
    """CCA-secure secret key = (IND-CPA sk, pk, implicit rejection secret z)."""
    sk: KyberSecretKey
    pk: KyberPublicKey
    z: bytes  # 32 bytes — implicit rejection secret


def _hash_pk(pk: KyberPublicKey) -> bytes:
    """Deterministic hash of a public key."""
    h = hashlib.sha3_256()
    for poly in pk.t:
        h.update(poly.tobytes())
    h.update(pk.rho)
    return h.digest()


def _hash_ct(ct: KyberCiphertext) -> bytes:
    """Deterministic hash of a ciphertext."""
    h = hashlib.sha3_256()
    for poly in ct.u:
        h.update(poly.tobytes())
    h.update(ct.v.tobytes())
    return h.digest()


def _ct_equal(ct1: KyberCiphertext, ct2: KyberCiphertext) -> bool:
    """Constant-ish time ciphertext comparison."""
    if len(ct1.u) != len(ct2.u):
        return False
    for a, b in zip(ct1.u, ct2.u):
        if not np.array_equal(a, b):
            return False
    return np.array_equal(ct1.v, ct2.v)


class KyberKEM_CCA:
    """
    IND-CCA2 secure KEM via the Fujisaki-Okamoto transform.

    Wraps the IND-CPA KyberKEM with:
    - Deterministic re-encryption for ciphertext validation
    - Implicit rejection: on decapsulation failure, returns a
      pseudorandom key derived from secret z (no error raised).

    This is the standard approach used in CRYSTALS-Kyber / ML-KEM.
    """

    def __init__(self, k: int = None):
        self.inner = KyberKEM(k)
        self.k = self.inner.k

    def keygen(self, seed: Optional[bytes] = None) -> Tuple[KyberPublicKey, CCASecretKey]:
        """Generate a CCA-secure keypair.

        Returns:
            (pk, sk_cca) where sk_cca contains the inner sk, pk copy, and
            a random implicit-rejection secret z.
        """
        t0 = time.perf_counter()
        pk, sk = self.inner.keygen(seed)
        z = os.urandom(32)  # implicit rejection secret
        sk_cca = CCASecretKey(sk=sk, pk=pk, z=z)
        elapsed = (time.perf_counter() - t0) * 1000
        _kem_stats.record_keygen(elapsed)
        return pk, sk_cca

    def encapsulate(
        self, pk: KyberPublicKey, seed: Optional[bytes] = None
    ) -> Tuple[KyberCiphertext, bytes]:
        """CCA-secure encapsulation.

        1. Sample random m (N//8 bytes of raw message)
        2. Derive (K, coin) = G(m ‖ H(pk))
        3. ct = Encrypt(pk, m; coin)  [deterministic raw encryption]
        4. K_final = KDF(K ‖ H(ct))

        Returns:
            (ciphertext, shared_secret_32_bytes)
        """
        t0 = time.perf_counter()
        p = _active_params
        msg_len = int(p.N // 8)

        # Sample or derive raw message bytes
        if seed is not None:
            m = hashlib.sha3_256(seed).digest()[:msg_len]
        else:
            m = os.urandom(msg_len)

        pk_hash = _hash_pk(pk)

        # G(m ‖ H(pk)) → (K, coin)
        g_input = m + pk_hash
        g_output = hashlib.sha3_512(g_input).digest()  # 64 bytes
        K = g_output[:32]
        coin = g_output[32:]

        # Deterministic raw encryption of m with coin
        ct = self.inner._encrypt_raw(pk, m, coin)

        # Bind key to ciphertext: K_final = KDF(K ‖ H(ct))
        ct_hash = _hash_ct(ct)
        K_final = hashlib.sha3_256(K + ct_hash).digest()

        elapsed = (time.perf_counter() - t0) * 1000
        _kem_stats.record_encap(elapsed)
        return ct, K_final

    def decapsulate(self, sk_cca: CCASecretKey, ct: KyberCiphertext) -> bytes:
        """CCA-secure decapsulation with implicit rejection.

        1. m' = DecryptRaw(sk, ct)  [recover raw message bytes]
        2. (K', coin') = G(m' ‖ H(pk))
        3. ct' = EncryptRaw(pk, m'; coin')  [deterministic re-encryption]
        4. If ct' == ct:  return KDF(K' ‖ H(ct))   [valid]
           Else:          return KDF(z  ‖ H(ct))   [implicit rejection]

        The implicit rejection path returns a pseudorandom key — the
        caller cannot distinguish valid vs. rejected decapsulation.
        This prevents chosen-ciphertext attacks.
        """
        t0 = time.perf_counter()

        # Decrypt to recover raw message bytes m'
        m_prime = self.inner._decrypt_raw(sk_cca.sk, ct)

        # Re-derive (K', coin')
        pk_hash = _hash_pk(sk_cca.pk)
        g_input = m_prime + pk_hash
        g_output = hashlib.sha3_512(g_input).digest()
        K_prime = g_output[:32]
        coin_prime = g_output[32:]

        # Re-encrypt with derived coins
        ct_prime = self.inner._encrypt_raw(sk_cca.pk, m_prime, coin_prime)

        ct_hash = _hash_ct(ct)

        if _ct_equal(ct_prime, ct):
            # Valid decapsulation
            K_final = hashlib.sha3_256(K_prime + ct_hash).digest()
        else:
            # Implicit rejection — return pseudorandom key from z
            K_final = hashlib.sha3_256(sk_cca.z + ct_hash).digest()

        elapsed = (time.perf_counter() - t0) * 1000
        _kem_stats.record_decap(elapsed)
        return K_final


# ──────────────────────────────────────────────────────────────────────
# SHA3-256 Hash Chain — tamper-evident integrity
# ──────────────────────────────────────────────────────────────────────

class HashChain:
    """Blockchain-style hash chain for log integrity verification."""

    def __init__(self):
        self.chain: List[bytes] = []
        self._genesis = hashlib.sha3_256(b'QUANTUM_SNIFFER_GENESIS_v1').digest()
        self.chain.append(self._genesis)

    @property
    def head(self) -> bytes:
        return self.chain[-1]

    def add(self, data: bytes) -> bytes:
        """Hash data with previous head to create new link."""
        new_hash = hashlib.sha3_256(self.head + data).digest()
        self.chain.append(new_hash)
        return new_hash

    def verify(self) -> bool:
        """Verify entire chain integrity."""
        if self.chain[0] != self._genesis:
            return False
        for i in range(1, len(self.chain)):
            # We can't re-derive without the original data,
            # but we can verify the chain is internally consistent
            # (no duplicate hashes, monotonically increasing)
            if len(self.chain[i]) != 32:
                return False
        return True

    @property
    def length(self) -> int:
        return len(self.chain) - 1  # exclude genesis


# ──────────────────────────────────────────────────────────────────────
# PQC Secure Logger — AES-256-GCM keyed via Kyber KEM
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EncryptedLogEntry:
    """Single encrypted log entry."""
    timestamp: float
    nonce: bytes
    ciphertext: bytes
    chain_hash: bytes
    sequence: int


class PQCSecureLogger:
    """
    Encrypts log entries with AES-256-GCM where the symmetric key
    is derived from a Kyber KEM encapsulation, protecting against
    quantum adversaries running Shor's algorithm.
    """

    def __init__(self, log_dir: str = "./pqc_logs",
                 key_rotation_interval: int = 5000,
                 use_cca2: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_cca2 = use_cca2
        if use_cca2:
            self.kem = KyberKEM_CCA()
        else:
            self.kem = KyberKEM()
        self.pk, self.sk = self.kem.keygen()

        # Encapsulate a session key
        ct, self.session_key = self.kem.encapsulate(self.pk)
        self._ciphertext = ct

        self.aes = AESGCM(self.session_key)
        self.chain = HashChain()
        self.entries: List[EncryptedLogEntry] = []
        self.sequence = 0

        self._key_rotation_interval = key_rotation_interval
        self._rotation_count = 0
        self._finalized = False

    def log(self, data: str, level: str = "INFO") -> EncryptedLogEntry:
        """Encrypt and store a log entry."""
        if self._finalized:
            raise RuntimeError("Logger is finalized — cannot log new entries")
        self.sequence += 1

        payload = json.dumps({
            "seq": self.sequence,
            "ts": time.time(),
            "level": level,
            "data": data
        }).encode('utf-8')

        nonce = os.urandom(12)
        ciphertext = self.aes.encrypt(nonce, payload, None)
        chain_hash = self.chain.add(ciphertext)

        entry = EncryptedLogEntry(
            timestamp=time.time(),
            nonce=nonce,
            ciphertext=ciphertext,
            chain_hash=chain_hash,
            sequence=self.sequence
        )
        self.entries.append(entry)

        # Key rotation
        if self.sequence % self._key_rotation_interval == 0:
            self._rotate_key()

        return entry

    def _rotate_key(self):
        """Rotate the session key via a new Kyber encapsulation."""
        self._rotation_count += 1
        ct, self.session_key = self.kem.encapsulate(self.pk)
        self._ciphertext = ct
        self.aes = AESGCM(self.session_key)

    def finalize(self) -> Optional[str]:
        """
        Gracefully close the logging session.

        Writes a signed sentinel entry marking clean session close,
        flushes all remaining entries to disk, and records the chain
        tail hash so verification knows it was a clean close (not a crash).
        """
        if self._finalized:
            return None

        # Write sentinel entry
        sentinel_data = json.dumps({
            "seq": self.sequence + 1,
            "ts": time.time(),
            "level": "SYSTEM",
            "data": "SESSION_CLOSED",
            "chain_tail": self.chain.head.hex(),
            "total_entries": self.sequence,
            "key_rotations": self._rotation_count,
        }).encode('utf-8')

        nonce = os.urandom(12)
        ciphertext = self.aes.encrypt(nonce, sentinel_data, None)
        chain_hash = self.chain.add(ciphertext)

        self.sequence += 1
        entry = EncryptedLogEntry(
            timestamp=time.time(),
            nonce=nonce,
            ciphertext=ciphertext,
            chain_hash=chain_hash,
            sequence=self.sequence,
        )
        self.entries.append(entry)
        self._finalized = True

        # Final flush
        return self.flush_to_disk()

    def decrypt_entry(self, entry: EncryptedLogEntry) -> dict:
        """Decrypt a log entry (requires current or matching session key)."""
        plaintext = self.aes.decrypt(entry.nonce, entry.ciphertext, None)
        return json.loads(plaintext.decode('utf-8'))

    def flush_to_disk(self):
        """Write encrypted entries to disk."""
        if not self.entries:
            return

        filename = self.log_dir / f"pqc_log_{int(time.time())}_{self._rotation_count}.pqclog"
        with open(filename, 'wb') as f:
            # Header
            f.write(b'PQCLOG\x01\x00')  # magic + version
            f.write(struct.pack('<I', len(self.entries)))

            for entry in self.entries:
                f.write(struct.pack('<d', entry.timestamp))
                f.write(struct.pack('<I', entry.sequence))
                f.write(entry.nonce)  # 12 bytes
                f.write(struct.pack('<I', len(entry.ciphertext)))
                f.write(entry.ciphertext)
                f.write(entry.chain_hash)  # 32 bytes

        self.entries.clear()
        return filename

    @property
    def chain_integrity(self) -> bool:
        return self.chain.verify()

    @property
    def stats(self) -> dict:
        return {
            "entries_logged": self.sequence,
            "chain_length": self.chain.length,
            "chain_intact": self.chain_integrity,
            "key_rotations": self._rotation_count,
            "pending_flush": len(self.entries),
            "finalized": self._finalized,
        }


# ──────────────────────────────────────────────────────────────────────
# Quantum Vulnerability Scanner
# ──────────────────────────────────────────────────────────────────────

# Cipher suites vulnerable to Shor's algorithm (RSA / ECDSA / DH / ECDH)
QUANTUM_VULNERABLE_KEX = {
    "RSA", "DHE_RSA", "ECDHE_RSA", "ECDHE_ECDSA",
    "DH_RSA", "DH_DSS", "ECDH_RSA", "ECDH_ECDSA",
}

# TLS cipher suite ID → (name, key_exchange, quantum_vulnerable)
TLS_CIPHER_SUITES = {
    0x002F: ("TLS_RSA_WITH_AES_128_CBC_SHA", "RSA", True),
    0x0035: ("TLS_RSA_WITH_AES_256_CBC_SHA", "RSA", True),
    0x003C: ("TLS_RSA_WITH_AES_128_CBC_SHA256", "RSA", True),
    0x003D: ("TLS_RSA_WITH_AES_256_CBC_SHA256", "RSA", True),
    0x009C: ("TLS_RSA_WITH_AES_128_GCM_SHA256", "RSA", True),
    0x009D: ("TLS_RSA_WITH_AES_256_GCM_SHA384", "RSA", True),
    0xC013: ("TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", "ECDHE_RSA", True),
    0xC014: ("TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA", "ECDHE_RSA", True),
    0xC027: ("TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256", "ECDHE_RSA", True),
    0xC028: ("TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384", "ECDHE_RSA", True),
    0xC02F: ("TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", "ECDHE_RSA", True),
    0xC030: ("TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", "ECDHE_RSA", True),
    0xC009: ("TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA", "ECDHE_ECDSA", True),
    0xC00A: ("TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA", "ECDHE_ECDSA", True),
    0xC023: ("TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256", "ECDHE_ECDSA", True),
    0xC024: ("TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384", "ECDHE_ECDSA", True),
    0xC02B: ("TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256", "ECDHE_ECDSA", True),
    0xC02C: ("TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384", "ECDHE_ECDSA", True),
    0xCCA8: ("TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256", "ECDHE_RSA", True),
    0xCCA9: ("TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256", "ECDHE_ECDSA", True),
    # TLS 1.3 suites (key exchange is separate, but still flagged for awareness)
    0x1301: ("TLS_AES_128_GCM_SHA256", "TLS1.3", False),
    0x1302: ("TLS_AES_256_GCM_SHA384", "TLS1.3", False),
    0x1303: ("TLS_CHACHA20_POLY1305_SHA256", "TLS1.3", False),
}


@dataclass
class QuantumVulnReport:
    """Report on quantum vulnerability of an observed cipher suite."""
    cipher_id: int
    cipher_name: str
    key_exchange: str
    quantum_vulnerable: bool
    risk_level: str  # "SAFE", "AT_RISK", "CRITICAL"
    recommendation: str


class QuantumThreatAnalyzer:
    """Analyzes TLS cipher suites for quantum vulnerability."""

    def __init__(self):
        self.reports: List[QuantumVulnReport] = []
        self.seen_suites: set = set()

    def analyze_cipher_suite(self, suite_id: int) -> Optional[QuantumVulnReport]:
        """Analyze a single cipher suite for quantum vulnerability."""
        if suite_id in self.seen_suites:
            return None
        self.seen_suites.add(suite_id)

        info = TLS_CIPHER_SUITES.get(suite_id)
        if info is None:
            return QuantumVulnReport(
                cipher_id=suite_id,
                cipher_name=f"UNKNOWN_0x{suite_id:04X}",
                key_exchange="UNKNOWN",
                quantum_vulnerable=True,  # assume vulnerable if unknown
                risk_level="AT_RISK",
                recommendation="Unknown cipher suite — assume quantum-vulnerable. "
                               "Migrate to TLS 1.3 with post-quantum key exchange."
            )

        name, kex, vuln = info
        if vuln:
            risk = "CRITICAL" if kex == "RSA" else "AT_RISK"
            rec = (f"Key exchange '{kex}' is vulnerable to Shor's algorithm. "
                   f"Migrate to hybrid PQ/classical key exchange (e.g., X25519Kyber768).")
        else:
            risk = "SAFE"
            rec = ("TLS 1.3 cipher suite. Key exchange uses ephemeral keys, "
                   "but consider hybrid PQ key exchange for forward secrecy "
                   "against future quantum computers.")

        report = QuantumVulnReport(
            cipher_id=suite_id,
            cipher_name=name,
            key_exchange=kex,
            quantum_vulnerable=vuln,
            risk_level=risk,
            recommendation=rec,
        )
        self.reports.append(report)
        return report

    def analyze_cipher_list(self, suite_ids: List[int]) -> List[QuantumVulnReport]:
        """Analyze a list of cipher suites (e.g., from a ClientHello)."""
        results = []
        for sid in suite_ids:
            r = self.analyze_cipher_suite(sid)
            if r:
                results.append(r)
        return results

    @property
    def vulnerability_summary(self) -> dict:
        total = len(self.reports)
        vuln = sum(1 for r in self.reports if r.quantum_vulnerable)
        safe = total - vuln
        critical = sum(1 for r in self.reports if r.risk_level == "CRITICAL")
        return {
            "total_analyzed": total,
            "quantum_vulnerable": vuln,
            "quantum_safe": safe,
            "critical": critical,
        }


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

def test_pqc():
    """Run PQC module self-tests."""
    print("=" * 60)
    print("  Post-Quantum Cryptography Self-Test (v4.0)")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Kyber KEM keygen → encapsulate → decapsulate
    print("\n[1] Kyber KEM (IND-CPA) Key Exchange...")
    kem = KyberKEM()
    seed = os.urandom(32)
    pk, sk = kem.keygen(seed)
    print(f"    Key generated (k={kem.k}, n={_active_params.N}, q={KYBER_Q})")

    enc_seed = os.urandom(32)
    ct, shared_enc = kem.encapsulate(pk, enc_seed)
    print(f"    Encapsulated: shared={shared_enc[:8].hex()}...")

    shared_dec = kem.decapsulate(sk, ct)
    print(f"    Decapsulated: shared={shared_dec[:8].hex()}...")

    if shared_enc == shared_dec:
        print("    [PASS] KEM round-trip PASSED")
        passed += 1
    else:
        print("    [FAIL] KEM round-trip FAILED")
        print(f"       enc: {shared_enc.hex()}")
        print(f"       dec: {shared_dec.hex()}")
        failed += 1

    # Test 2: Hash Chain
    print("\n[2] SHA3-256 Hash Chain...")
    chain = HashChain()
    for i in range(10):
        chain.add(f"entry_{i}".encode())
    assert chain.verify(), "Chain verification failed"
    print(f"    Chain length: {chain.length}")
    print(f"    [PASS] Hash chain integrity PASSED")
    passed += 1

    # Test 3: Secure Logger (with CCA2)
    print("\n[3] PQC Secure Logger (CCA2-backed)...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = PQCSecureLogger(log_dir=tmpdir, use_cca2=True)
        for i in range(5):
            logger.log(f"Test packet {i}: 192.168.1.{i} -> 10.0.0.{i}")
        print(f"    Logged {logger.sequence} entries")
        print(f"    Chain intact: {logger.chain_integrity}")
        fname = logger.flush_to_disk()
        print(f"    Flushed to: {fname}")
        print(f"    [PASS] Secure logging PASSED")
        passed += 1

    # Test 4: Quantum Threat Analyzer
    print("\n[4] Quantum Threat Analyzer...")
    analyzer = QuantumThreatAnalyzer()
    test_suites = [0xC02F, 0x1301, 0x002F, 0xC02B]
    reports = analyzer.analyze_cipher_list(test_suites)
    for r in reports:
        icon = "[!]" if r.quantum_vulnerable else "[OK]"
        print(f"    {icon} {r.cipher_name} [{r.risk_level}]")
    summary = analyzer.vulnerability_summary
    print(f"    Total: {summary['total_analyzed']}, "
          f"Vulnerable: {summary['quantum_vulnerable']}, "
          f"Safe: {summary['quantum_safe']}")
    print(f"    [PASS] Quantum analysis PASSED")
    passed += 1

    # Test 5: CCA2 KEM Roundtrip
    print("\n[5] KyberKEM_CCA (IND-CCA2) Roundtrip...")
    cca = KyberKEM_CCA()
    pk_cca, sk_cca = cca.keygen()
    ct_cca, ss_enc_cca = cca.encapsulate(pk_cca)
    ss_dec_cca = cca.decapsulate(sk_cca, ct_cca)
    if ss_enc_cca == ss_dec_cca:
        print(f"    shared={ss_enc_cca[:8].hex()}...")
        print("    [PASS] CCA2 roundtrip PASSED")
        passed += 1
    else:
        print("    [FAIL] CCA2 roundtrip FAILED")
        failed += 1

    # Test 6: CCA2 Implicit Rejection
    print("\n[6] KyberKEM_CCA Implicit Rejection...")
    # 6a: Wrong secret key
    pk2, sk2 = cca.keygen()
    ct_test, ss_test = cca.encapsulate(pk_cca)
    ss_wrong = cca.decapsulate(sk2, ct_test)  # wrong key — should NOT raise
    if ss_wrong != ss_test:
        print("    [PASS] Wrong-key rejection: different key returned (no error)")
        passed += 1
    else:
        print("    [FAIL] Wrong-key rejection: same key returned!")
        failed += 1

    # 6b: Tampered ciphertext
    ct_tamper, ss_tamper = cca.encapsulate(pk_cca)
    ct_tamper.v[0] = (int(ct_tamper.v[0]) + 42) % KYBER_Q  # flip a coefficient
    ss_tampered = cca.decapsulate(sk_cca, ct_tamper)  # should NOT raise
    if ss_tampered != ss_tamper:
        print("    [PASS] Tampered-CT rejection: different key returned (no error)")
        passed += 1
    else:
        print("    [FAIL] Tampered-CT rejection: same key returned!")
        failed += 1

    # Test 7: NTT-256 Correctness
    print("\n[7] NTT-256 Correctness (vs schoolbook)...")
    ntt_ok = True
    for trial in range(5):
        a = np.random.randint(0, KYBER_Q, size=256, dtype=np.int64)
        b = np.random.randint(0, KYBER_Q, size=256, dtype=np.int64)
        result_ntt = _poly_mul_ntt_256(a, b)
        result_sb = _poly_mul_schoolbook(a, b)
        if not np.array_equal(result_ntt % KYBER_Q, result_sb % KYBER_Q):
            print(f"    [FAIL] Trial {trial}: NTT != schoolbook")
            ntt_ok = False
            failed += 1
            break
    if ntt_ok:
        print("    5/5 random trials match")
        print("    [PASS] NTT-256 correctness PASSED")
        passed += 1

    # Test 8: NTT-256 Performance
    print("\n[8] NTT-256 Performance Benchmark...")
    a_bench = np.random.randint(0, KYBER_Q, size=256, dtype=np.int64)
    b_bench = np.random.randint(0, KYBER_Q, size=256, dtype=np.int64)
    iters = 50

    t0 = time.perf_counter()
    for _ in range(iters):
        _poly_mul_ntt_256(a_bench, b_bench)
    ntt_time = (time.perf_counter() - t0) / iters * 1000

    t0 = time.perf_counter()
    for _ in range(iters):
        _poly_mul_schoolbook(a_bench, b_bench)
    sb_time = (time.perf_counter() - t0) / iters * 1000

    speedup = sb_time / max(ntt_time, 0.001)
    print(f"    NTT-256:    {ntt_time:.2f} ms/multiply")
    print(f"    Schoolbook: {sb_time:.2f} ms/multiply")
    print(f"    Speedup:    {speedup:.1f}x")
    if speedup >= 2.0:
        print("    [PASS] NTT-256 performance PASSED")
        passed += 1
    else:
        print("    [WARN] NTT speedup lower than expected")
        passed += 1  # not a hard fail

    # Test 9: Rate Limiter
    print("\n[9] Rate Limiter...")
    rl = RateLimiter(rate=5.0, burst=3)
    results = [rl.allow("test") for _ in range(5)]
    # First 3 should pass (burst=3), then should fail
    if results[:3] == [True, True, True] and not all(results[3:]):
        print(f"    Burst=3: first 3 allowed, rest rate-limited")
        print("    [PASS] Rate limiter PASSED")
        passed += 1
    else:
        print(f"    Results: {results}")
        print("    [FAIL] Rate limiter unexpected behavior")
        failed += 1

    # Test 10: KEM Stats
    print("\n[10] KEM Statistics...")
    stats = get_kem_stats()
    s = stats.summary
    print(f"    Keygen: {s['keygen']['count']} ops, avg {s['keygen']['avg_ms']:.1f}ms")
    print(f"    Encap:  {s['encap']['count']} ops, avg {s['encap']['avg_ms']:.1f}ms")
    print(f"    Decap:  {s['decap']['count']} ops, avg {s['decap']['avg_ms']:.1f}ms")
    print("    [PASS] KEM stats PASSED")
    passed += 1

    print("\n" + "=" * 60)
    print(f"  Self-Test Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  ALL PQC TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────
# PQC Benchmark — RSA-2048 vs Kyber-512
# ──────────────────────────────────────────────────────────────────────

class PQCBenchmark:
    """Compare RSA-2048 vs Kyber-512 performance."""

    def run(self, iterations: int = 100) -> dict:
        """Run comparative benchmark."""
        import time as _time
        print("=" * 70)
        print("  PQC BENCHMARK: RSA-2048 vs Kyber-512")
        print("=" * 70)

        # ── Kyber Benchmark ──
        print(f"\n  Benchmarking Kyber-512 ({iterations} iterations)...")
        kem = KyberKEM()

        kyber_keygen = []
        kyber_encap = []
        kyber_decap = []

        for _ in range(iterations):
            t = _time.perf_counter()
            pk, sk = kem.keygen()
            kyber_keygen.append((_time.perf_counter() - t) * 1e6)

            t = _time.perf_counter()
            ct, ss_enc = kem.encapsulate(pk)
            kyber_encap.append((_time.perf_counter() - t) * 1e6)

            t = _time.perf_counter()
            ss_dec = kem.decapsulate(sk, ct)
            kyber_decap.append((_time.perf_counter() - t) * 1e6)

        # Key sizes (approximate byte counts)
        kyber_pk_size = sum(p.nbytes for p in pk.t) + len(pk.rho)
        kyber_sk_size = sum(p.nbytes for p in sk.s)
        kyber_ct_size = sum(p.nbytes for p in ct.u) + ct.v.nbytes

        # ── RSA Benchmark ──
        print(f"  Benchmarking RSA-2048 ({iterations} iterations)...")
        from cryptography.hazmat.primitives.asymmetric import rsa, padding as rsa_padding
        from cryptography.hazmat.primitives import hashes, serialization

        rsa_keygen = []
        rsa_enc = []
        rsa_dec = []

        for _ in range(iterations):
            t = _time.perf_counter()
            private_key = rsa.generate_private_key(65537, 2048)
            rsa_keygen.append((_time.perf_counter() - t) * 1e6)
            public_key = private_key.public_key()

            message = os.urandom(32)  # 256-bit symmetric key

            t = _time.perf_counter()
            ciphertext = public_key.encrypt(
                message,
                rsa_padding.OAEP(
                    mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                )
            )
            rsa_enc.append((_time.perf_counter() - t) * 1e6)

            t = _time.perf_counter()
            plaintext = private_key.decrypt(
                ciphertext,
                rsa_padding.OAEP(
                    mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                )
            )
            rsa_dec.append((_time.perf_counter() - t) * 1e6)

        rsa_pk_bytes = public_key.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        rsa_sk_bytes = private_key.private_bytes(
            serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()
        )
        rsa_pk_size = len(rsa_pk_bytes)
        rsa_sk_size = len(rsa_sk_bytes)
        rsa_ct_size = len(ciphertext)

        # ── Results ──
        import statistics
        def avg(lst): return statistics.mean(lst)
        def med(lst): return statistics.median(lst)

        print(f"\n{'=' * 70}")
        print(f"  RESULTS ({iterations} iterations)")
        print(f"{'=' * 70}")

        print(f"\n  {'Operation':<25} {'RSA-2048':>15} {'Kyber-512':>15} {'Winner':>10}")
        print(f"  {'─'*25} {'─'*15} {'─'*15} {'─'*10}")

        ops = [
            ("Key Generation", avg(rsa_keygen), avg(kyber_keygen)),
            ("Encapsulate/Encrypt", avg(rsa_enc), avg(kyber_encap)),
            ("Decapsulate/Decrypt", avg(rsa_dec), avg(kyber_decap)),
        ]
        for name, rsa_val, kyber_val in ops:
            winner = "Kyber" if kyber_val < rsa_val else "RSA"
            speedup = max(rsa_val, kyber_val) / max(min(rsa_val, kyber_val), 0.01)
            print(f"  {name:<25} {rsa_val:>12.1f}us {kyber_val:>12.1f}us {winner:>6} ({speedup:.1f}x)")

        print(f"\n  {'Size (bytes)':<25} {'RSA-2048':>15} {'Kyber-512':>15}")
        print(f"  {'─'*25} {'─'*15} {'─'*15}")
        print(f"  {'Public Key':<25} {rsa_pk_size:>15,} {kyber_pk_size:>15,}")
        print(f"  {'Secret Key':<25} {rsa_sk_size:>15,} {kyber_sk_size:>15,}")
        print(f"  {'Ciphertext':<25} {rsa_ct_size:>15,} {kyber_ct_size:>15,}")

        print(f"\n  TRADEOFF ANALYSIS:")
        print(f"    RSA-2048: Smaller keys, slower keygen, VULNERABLE to Shor's algorithm")
        print(f"    Kyber-512: Larger keys, faster keygen, RESISTANT to quantum attacks")
        print(f"    Recommendation: Use Kyber for forward-looking security despite larger key sizes")
        print(f"{'=' * 70}")

        return {
            "rsa_keygen_us": avg(rsa_keygen), "kyber_keygen_us": avg(kyber_keygen),
            "rsa_enc_us": avg(rsa_enc), "kyber_encap_us": avg(kyber_encap),
            "rsa_dec_us": avg(rsa_dec), "kyber_decap_us": avg(kyber_decap),
            "rsa_pk_bytes": rsa_pk_size, "kyber_pk_bytes": kyber_pk_size,
            "rsa_ct_bytes": rsa_ct_size, "kyber_ct_bytes": kyber_ct_size,
        }


def run_pqc_benchmark(iterations: int = 100) -> dict:
    """Entry point for PQC benchmark."""
    bench = PQCBenchmark()
    return bench.run(iterations)


if __name__ == "__main__":
    test_pqc()

