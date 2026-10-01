"""Minimal NumPy MLP (SiLU) with manual back-propagation and Adam. No deep-learning framework required."""
import numpy as np


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self._m = [np.zeros_like(p) for p in self.W + self.b]
        self._v = [np.zeros_like(p) for p in self.W + self.b]
        self._t = 0

    def forward(self, x, cache=False):
        self._cache = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = x @ W + b
            if i < len(self.W) - 1:
                s = 1 / (1 + np.exp(-np.clip(z, -30, 30))); x = z * s
                if cache: self._cache += [z, s, x]
            else:
                x = z
        return x

    __call__ = forward

    def step(self, grad, lr, b1=0.9, b2=0.999):
        """Back-propagate dLoss/dOutput through the cached forward pass and apply one Adam update."""
        L = len(self.W); gW = [None] * L; gb = [None] * L; g = grad
        for i in reversed(range(L)):
            xin = self._cache[0] if i == 0 else self._cache[3 * i]
            gW[i] = xin.T @ g; gb[i] = g.sum(0)
            if i > 0:
                g = g @ self.W[i].T
                z, s = self._cache[3 * i - 2], self._cache[3 * i - 1]
                g = g * (s * (1 + z * (1 - s)))
        self._t += 1
        for j, (p, gr) in enumerate(zip(self.W + self.b, gW + gb)):
            self._m[j] = b1 * self._m[j] + (1 - b1) * gr
            self._v[j] = b2 * self._v[j] + (1 - b2) * gr ** 2
            mh = self._m[j] / (1 - b1 ** self._t); vh = self._v[j] / (1 - b2 ** self._t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)

    def n_params(self):
        return sum(w.size for w in self.W) + sum(b.size for b in self.b)

    def state(self, prefix):
        d = {f'{prefix}_W{i}': w for i, w in enumerate(self.W)}
        d.update({f'{prefix}_b{i}': b for i, b in enumerate(self.b)}); return d

    @classmethod
    def from_state(cls, d, prefix):
        n = len([k for k in d if k.startswith(f'{prefix}_W')])
        W = [d[f'{prefix}_W{i}'] for i in range(n)]
        net = cls([W[0].shape[0]] + [w.shape[1] for w in W], np.random.default_rng(0))
        net.W = [w.copy() for w in W]; net.b = [d[f'{prefix}_b{i}'].copy() for i in range(n)]
        return net
