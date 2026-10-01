"""Beam-training schemes: DiffBeam (Algorithm 1) and baselines (Sec. IV)."""
import numpy as np
from .config import N, S_POLAR, S_HIER, V_MIN, V_MAX
from .channel import steer, probe, true_gain, rate, gen_channels
from .codebooks import dft_codebook, polar_codebook

_UG, _FF = dft_codebook()
_VH = np.linspace(V_MIN, V_MAX, S_HIER)
_POL = None


def _polar():
    global _POL
    if _POL is None:
        _POL = polar_codebook(S_POLAR)
    return _POL


def select_by_probing(u, r, h, snr, K, rng):
    """Stage 2: probe K candidate focusing beams per user and keep the strongest."""
    Wc = steer(u, r).reshape(h.shape[0], K, N)
    p = probe(Wc, h, snr, rng)
    return Wc[np.arange(h.shape[0]), p.argmax(1)]


def perfect_csi(h, snr, rng=None, **_):
    return rate(np.sum(np.abs(h) ** 2, 1), snr)


def exhaustive_polar(h, snr, rng, **_):
    W = _polar(); p = probe(W, h, snr, rng)
    return rate(true_gain(W[p.argmax(1)], h), snr)


def far_field_dft(h, snr, rng, **_):
    p = probe(_FF, h, snr, rng)
    return rate(true_gain(_FF[p.argmax(1)], h), snr)


def two_stage_hierarchical(h, snr, rng, **_):
    B = h.shape[0]; p = probe(_FF, h, snr, rng); ub = _UG[p.argmax(1)]
    Wc = steer(np.repeat(ub, S_HIER), np.tile(1 / _VH, B)).reshape(B, S_HIER, N)
    q = probe(Wc, h, snr, rng)
    return rate(true_gain(Wc[np.arange(B), q.argmax(1)], h), snr)


def dnn_regression(h, snr, rng, model, **_):
    x, ctx = model.regress(probe(model.W, h, snr, rng))
    u, r = model.cond.from_x(x, ctx)
    return rate(true_gain(steer(u, r), h), snr)


def dnn_refinement(h, snr, rng, model, K=8, jitter=None, **_):
    """DNN estimate + (K-1) Gaussian perturbations (std = validation RMSE), verified as in Stage 2."""
    B = h.shape[0]; x, ctx = model.regress(probe(model.W, h, snr, rng))
    xs = np.repeat(x, K, 0) + rng.standard_normal((B * K, 2)) * jitter
    xs.reshape(B, K, 2)[:, 0] = x
    u, r = model.cond.from_x(xs, ctx, K)
    return rate(true_gain(select_by_probing(u, r, h, snr, K, rng), h), snr)


def diffbeam(h, snr, rng, model, K=8, **_):
    """Algorithm 1: sensing sweep -> conditioning -> K posterior samples -> probing -> selection."""
    u, r = model.sample_candidates(probe(model.W, h, snr, rng), K, rng)
    if K == 1:
        return rate(true_gain(steer(u, r), h), snr)
    return rate(true_gain(select_by_probing(u, r, h, snr, K, rng), h), snr)


def dnn_jitter(model, rng, n=3000):
    """Per-dimension validation RMSE of the DNN regressor (used by dnn_refinement)."""
    h, u, v = gen_channels(n, rng); snr = 10 ** (rng.uniform(-15, 15, n) / 10)
    x, ctx = model.regress(probe(model.W, h, snr, rng))
    return np.sqrt(np.mean((x - model.cond.to_x(u, v, ctx)) ** 2, 0))


SCHEMES = {
    'Perfect CSI': perfect_csi,
    'Exhaustive polar': exhaustive_polar,
    'Far-field DFT': far_field_dft,
    'Two-stage hierarchical': two_stage_hierarchical,
    'DNN regression': dnn_regression,
    'DNN + refinement': dnn_refinement,
    'DiffBeam': diffbeam,
    'DiffBeam (K=1)': lambda h, snr, rng, model, **kw: diffbeam(h, snr, rng, model, K=1),
}

OVERHEAD = {'Exhaustive polar': N * S_POLAR, 'Two-stage hierarchical': 256 + S_HIER, 'Far-field DFT': 256,
            'DNN regression': 56, 'DNN + refinement': 64, 'DiffBeam (K=1)': 56, 'DiffBeam': 64}
