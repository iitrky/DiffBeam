"""Near-field XL-MIMO channel model, beam probing and rate metrics (Sec. II)."""
import numpy as np
from .config import N, LAM, DELTA, U_MAX, R_MIN, R_MAX, L_NLOS, R_NLOS_MAX


def steer(u, r):
    """Exact spherical-wave steering vectors, eqs. (1)-(2). u = sin(theta). Returns (B, N), unit norm."""
    u = np.atleast_1d(u)[:, None]; r = np.atleast_1d(r)[:, None]
    rn = np.sqrt(r ** 2 + DELTA[None] ** 2 - 2 * r * DELTA[None] * u)
    return np.exp(-1j * 2 * np.pi / LAM * (rn - r)) / np.sqrt(N)


def steer_fresnel(u, curv):
    """Fresnel-approximation codeword with wavefront curvature curv = 1/r (negative => diverging), eq. (7)."""
    u = np.atleast_1d(u)[:, None]; curv = np.atleast_1d(curv)[:, None]
    ph = -2 * np.pi / LAM * (-DELTA[None] * u + DELTA[None] ** 2 * (1 - u ** 2) * curv / 2)
    return np.exp(1j * ph) / np.sqrt(N)


def gen_channels(B, rng, K_db=10.0, r_range=(R_MIN, R_MAX)):
    """LoS + L near-field NLoS paths, eq. (3). Returns h (B,N) with E||h||^2 ~ N, and LoS (u, v=1/r)."""
    u0 = rng.uniform(-U_MAX, U_MAX, B)
    v0 = rng.uniform(1 / r_range[1], 1 / r_range[0], B)
    h = steer(u0, 1 / v0) * np.exp(1j * rng.uniform(0, 2 * np.pi, B))[:, None]
    pnl = 10 ** (-np.asarray(K_db) / 10) / L_NLOS
    for _ in range(L_NLOS):
        ul = rng.uniform(-U_MAX, U_MAX, B); rl = rng.uniform(R_MIN, R_NLOS_MAX, B)
        g = np.sqrt(pnl / 2) * (rng.standard_normal(B) + 1j * rng.standard_normal(B))
        h = h + g[:, None] * steer(ul, rl)
    return np.sqrt(N) * h, u0, v0


def probe(W, h, snr_lin, rng):
    """Noisy received power for each probing beam, eq. (4). W: (T,N) shared or (B,T,N) per user."""
    s = h @ W.conj().T if W.ndim == 2 else np.einsum('bkn,bn->bk', W.conj(), h)
    n = (rng.standard_normal(s.shape) + 1j * rng.standard_normal(s.shape)) / np.sqrt(2)
    snr = np.broadcast_to(np.atleast_1d(snr_lin), (h.shape[0],))[:, None]
    return np.abs(np.sqrt(snr) * s + n) ** 2


def true_gain(w, h):
    return np.abs(np.sum(w.conj() * h, axis=-1)) ** 2


def rate(gain, snr_lin):
    """Achievable rate, eq. (5)."""
    return np.log2(1 + snr_lin * gain)
