"""DiffBeam model: sensing codebook + conditioner + diffusion denoiser + DNN regression baseline."""
import json
import numpy as np
from .config import M_SENSE, HIDDEN
from .codebooks import sensing_codebook
from .conditioning import Conditioner
from .nn import MLP
from . import diffusion


class DiffBeamModel:
    def __init__(self, M=M_SENSE, sensing='defocus', conditioning='palw', hidden=HIDDEN, seed=1):
        self.M, self.sensing, self.conditioning = M, sensing, conditioning
        self.W = sensing_codebook(M, sensing)
        self.cond = Conditioner(conditioning)
        cd = self.cond.cond_dim(M); rng = np.random.default_rng(seed)
        self.denoiser = MLP([2 + cd + 16, hidden, hidden, hidden, 2], rng)
        self.regressor = MLP([cd, hidden, hidden, hidden, 2], rng)   # DNN regression baseline
        self.meta = {}

    def sample_candidates(self, p, K, rng):
        c, ctx = self.cond(p)
        return self.cond.from_x(diffusion.sample(self.denoiser, c, K, rng), ctx, K)

    def regress(self, p):
        c, ctx = self.cond(p)
        return self.regressor(c), ctx

    def save(self, path, extra=None):
        meta = dict(M=self.M, sensing=self.sensing, conditioning=self.conditioning, **(extra or {}))
        np.savez_compressed(path, meta=json.dumps(meta), **self.denoiser.state('den'), **self.regressor.state('reg'))

    @classmethod
    def load(cls, path):
        d = dict(np.load(path, allow_pickle=False))
        meta = json.loads(str(d.pop('meta')))
        m = cls(meta['M'], meta['sensing'], meta['conditioning'])
        m.denoiser = MLP.from_state(d, 'den'); m.regressor = MLP.from_state(d, 'reg'); m.meta = meta
        return m
