"""Conditional DDPM: cosine schedule, eps-prediction loss (eq. 11) and ancestral sampling (Alg. 1)."""
import numpy as np
from .config import T_DIFF

_t = np.arange(T_DIFF + 1) / T_DIFF
_ab = np.cos((_t + 0.008) / 1.008 * np.pi / 2) ** 2; _ab /= _ab[0]
BETA = np.clip(1 - _ab[1:] / _ab[:-1], 1e-5, 0.5)
ALPHA = 1 - BETA
ABAR = np.cumprod(ALPHA)


def time_embedding(t):
    f = np.exp(np.linspace(0, np.log(100), 8))
    a = (np.asarray(t)[:, None] / T_DIFF) * f[None]
    return np.concatenate([np.sin(a), np.cos(a)], 1)


def train_step(net, x0, c, rng, lr):
    """One epsilon-prediction training step (eq. 10-11); returns the batch loss."""
    B = x0.shape[0]
    t = rng.integers(0, T_DIFF, B); eps = rng.standard_normal(x0.shape)
    xt = np.sqrt(ABAR[t])[:, None] * x0 + np.sqrt(1 - ABAR[t])[:, None] * eps
    out = net.forward(np.concatenate([xt, c, time_embedding(t)], 1), cache=True)
    net.step(2 * (out - eps) / out.size, lr)
    return float(np.mean((out - eps) ** 2))


def sample(net, c, K, rng, clip=1.05):
    """Draw K posterior samples per conditioning row. Output (B*K, 2), grouped by row."""
    B = c.shape[0]; cc = np.repeat(c, K, 0)
    x = rng.standard_normal((B * K, 2))
    for t in reversed(range(T_DIFF)):
        eps = net.forward(np.concatenate([x, cc, time_embedding(np.full(B * K, t))], 1))
        x0 = np.clip((x - np.sqrt(1 - ABAR[t]) * eps) / np.sqrt(ABAR[t]), -clip, clip)
        if t > 0:
            ap = ABAR[t - 1]
            mean = (np.sqrt(ap) * BETA[t] / (1 - ABAR[t])) * x0 + (np.sqrt(ALPHA[t]) * (1 - ap) / (1 - ABAR[t])) * x
            x = mean + np.sqrt(BETA[t] * (1 - ap) / (1 - ABAR[t])) * rng.standard_normal(x.shape)
        else:
            x = x0
    return x
