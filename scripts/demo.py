"""Minimal demo: train-free inference with the pretrained checkpoint on a few random users."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffbeam.model import DiffBeamModel
from diffbeam.channel import gen_channels
from diffbeam import schemes

rng = np.random.default_rng(0)
model = DiffBeamModel.load(os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'diffbeam_defocus_palw.npz'))
h, u, v = gen_channels(300, rng)
snr = 1.0   # 0 dB per antenna
for name in ['Perfect CSI', 'Exhaustive polar', 'Two-stage hierarchical', 'Far-field DFT', 'DiffBeam']:
    kw = dict(model=model, K=8)
    r = np.mean(schemes.SCHEMES[name](h, snr, rng, **kw))
    print(f'{name:24s} pilots={schemes.OVERHEAD.get(name, "-")!s:>5}  rate={r:5.2f} bit/s/Hz')
