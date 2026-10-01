"""Train the DiffBeam denoiser and the DNN regression baseline on synthetic near-field channels.

Examples
  python scripts/train.py                                   # proposed model (defocus + PALW)
  python scripts/train.py --sensing dft --out checkpoints/diffbeam_dft_palw.npz
  python scripts/train.py --conditioning global --iters 12000 --out checkpoints/diffbeam_defocus_global.npz
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffbeam.model import DiffBeamModel
from diffbeam.channel import gen_channels, probe
from diffbeam import diffusion

ap = argparse.ArgumentParser()
ap.add_argument('--M', type=int, default=56)
ap.add_argument('--sensing', default='defocus', choices=['defocus', 'dft'])
ap.add_argument('--conditioning', default='palw', choices=['palw', 'global'])
ap.add_argument('--iters', type=int, default=15000)
ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--snr_min', type=float, default=-15); ap.add_argument('--snr_max', type=float, default=15)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--out', default='checkpoints/diffbeam_defocus_palw.npz')
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
m = DiffBeamModel(a.M, a.sensing, a.conditioning, seed=a.seed)
t0 = time.time(); ld = lr_ = 0.0
for it in range(a.iters):
    h, u, v = gen_channels(a.batch, rng, K_db=rng.uniform(5, 15))
    snr = 10 ** (rng.uniform(a.snr_min, a.snr_max, a.batch) / 10)
    c, ctx = m.cond(probe(m.W, h, snr, rng)); x0 = m.cond.to_x(u, v, ctx)
    lr = a.lr * 0.5 * (1 + np.cos(np.pi * it / a.iters)) + 2e-5          # cosine decay
    ld = 0.99 * ld + 0.01 * diffusion.train_step(m.denoiser, x0, c, rng, lr)
    o = m.regressor.forward(c, cache=True)
    m.regressor.step(2 * (o - x0) / o.size, lr); lr_ = 0.99 * lr_ + 0.01 * float(np.mean((o - x0) ** 2))
    if it % 1000 == 0 or it == a.iters - 1:
        print(f'iter {it:6d}  diffusion loss {ld:.4f}  DNN mse {lr_:.4f}  [{time.time() - t0:.0f}s]', flush=True)
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
m.save(a.out, extra=dict(iters=a.iters, seed=a.seed))
print('saved', a.out)
