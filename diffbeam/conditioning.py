"""Conditioning vectors and target parameterisations.

'palw'   : peak-anchored local-window conditioning, eqs. (8)-(9) (proposed)
'global' : full normalised power profile + absolute target (ablation 'w/o PALW')
"""
import numpy as np
from .config import U_MAX, V_MIN, V_MAX, PALW_WIN, PALW_SCALE


def _v_norm(v): return 2 * (v - V_MIN) / (V_MAX - V_MIN) - 1
def _v_denorm(x): return (x + 1) / 2 * (V_MAX - V_MIN) + V_MIN


def global_features(p):
    p = p / (p.max(axis=1, keepdims=True) + 1e-12)
    return np.log10(p + 1e-3) / 3.0 + 0.5


def palw(p):
    """Returns conditioning vector c (B, 2W+3), anchor u_pk (B,), grid spacing g."""
    M = p.shape[1]; ug = np.linspace(-U_MAX, U_MAX, M); g = ug[1] - ug[0]
    k = p.argmax(1); pm = p.max(1, keepdims=True)
    pad = np.pad(p / pm, ((0, 0), (PALW_WIN, PALW_WIN)))
    win = np.take_along_axis(pad, k[:, None] + np.arange(2 * PALW_WIN + 1)[None], 1)
    c = np.concatenate([np.log10(win + 1e-3) / 3 + 0.5, (ug[k] / U_MAX)[:, None], np.log10(pm) / 4], 1)
    return c, ug[k], g


class Conditioner:
    def __init__(self, mode='palw'):
        assert mode in ('palw', 'global'); self.mode = mode

    def cond_dim(self, M):
        return 2 * PALW_WIN + 3 if self.mode == 'palw' else M

    def __call__(self, p):
        """Returns (c, ctx); ctx is needed to map targets back to (u, r)."""
        if self.mode == 'palw':
            c, upk, g = palw(p); return c, (upk, g)
        return global_features(p), None

    def to_x(self, u, v, ctx):
        if self.mode == 'palw':
            upk, g = ctx
            return np.clip(np.stack([(u - upk) / (PALW_SCALE * g), _v_norm(v)], 1), -1, 1)
        return np.stack([u / U_MAX, _v_norm(v)], 1)

    def from_x(self, x, ctx, K=1):
        """Map normalised samples (B*K, 2) back to (u, r)."""
        x = np.clip(x, -1, 1)
        if self.mode == 'palw':
            upk, g = ctx
            u = np.clip(np.repeat(upk, K) + x[:, 0] * PALW_SCALE * g, -1, 1)
        else:
            u = x[:, 0] * U_MAX
        return u, 1 / _v_denorm(x[:, 1])
