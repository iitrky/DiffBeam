"""Reproduce the main experiments of the paper (Figs. 3-5, Table III, runtime).

  python scripts/evaluate.py                 # full run  -> results/results_main.json
  python scripts/evaluate.py --quick         # smoke test (small sample sizes)
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffbeam.model import DiffBeamModel
from diffbeam.channel import gen_channels, probe
from diffbeam import schemes, diffusion

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', default='checkpoints/diffbeam_defocus_palw.npz')
ap.add_argument('--out', default='results/results_main.json')
ap.add_argument('--K', type=int, default=8)
ap.add_argument('--seed', type=int, default=123)
ap.add_argument('--quick', action='store_true')
a = ap.parse_args()

rng = np.random.default_rng(a.seed)
model = DiffBeamModel.load(a.ckpt)
JIT = schemes.dnn_jitter(model, rng)
B = 150 if a.quick else 800
Bk = 120 if a.quick else 600


def run(h, snr, K=a.K, only=None):
    out = {}
    for name, fn in schemes.SCHEMES.items():
        if only and name not in only: continue
        out[name] = float(np.mean(fn(h, snr, rng, model=model, K=K, jitter=JIT)))
    return out


R = {'config': vars(a), 'dnn_jitter': JIT.tolist(), 'overhead': schemes.OVERHEAD}
# Exp. 1: rate vs SNR (Fig. 3)
R['snr'] = [-15, -10, -5, 0, 5, 10, 15]; R['vs_snr'] = {}
for s in R['snr']:
    h, _, _ = gen_channels(B, rng); res = run(h, 10 ** (s / 10))
    for k, v in res.items(): R['vs_snr'].setdefault(k, []).append(v)
    print(f'SNR {s:4d} dB', {k: round(v, 2) for k, v in res.items()}, flush=True)
# Exp. 2: rate vs K (Fig. 4)
R['Ks'] = [1, 2, 4, 8, 16, 32]; R['vs_K'] = {}
for s in [-10, 0]:
    h, _, _ = gen_channels(Bk, rng)
    for K in R['Ks']:
        res = run(h, 10 ** (s / 10), K=K, only=['DiffBeam', 'DNN + refinement', 'Perfect CSI'])
        for k, v in res.items(): R['vs_K'].setdefault(f'{k}@{s}', []).append(v)
    print('K sweep', s, 'dB  DiffBeam', np.round(R['vs_K'][f'DiffBeam@{s}'], 2), flush=True)
# Exp. 3: rate vs distance at 0 dB (Fig. 5)
R['dbins'] = [(4, 6), (6, 9), (9, 13), (13, 19), (19, 30)]; R['vs_dist'] = {}
for lo, hi in R['dbins']:
    h, _, _ = gen_channels(Bk, rng, r_range=(lo, hi)); res = run(h, 1.0)
    for k, v in res.items(): R['vs_dist'].setdefault(k, []).append(v)
    print(f'dist {lo}-{hi} m', {k: round(v, 2) for k, v in res.items()}, flush=True)
# Exp. 4: Rician factor robustness at 0 dB (Table III)
R['vs_rician'] = {}
for kd in [0, 5, 10, 20]:
    h, _, _ = gen_channels(B, rng, K_db=kd); R['vs_rician'][kd] = run(h, 1.0)
    print(f'Rician {kd} dB', {k: round(v, 2) for k, v in R['vs_rician'][kd].items()}, flush=True)
# Exp. 5: runtime
h, _, _ = gen_channels(500, rng); c, _ = model.cond(probe(model.W, h, 1.0, rng))
t0 = time.time(); diffusion.sample(model.denoiser, c, a.K, rng); td = (time.time() - t0) / 500
t0 = time.time(); model.regressor(c); tr = (time.time() - t0) / 500
R['runtime_ms'] = {'DiffBeam': td * 1e3, 'DNN': tr * 1e3}
R['params'] = {'DiffBeam': model.denoiser.n_params(), 'DNN': model.regressor.n_params()}
print('runtime (ms/user):', R['runtime_ms'], 'params:', R['params'])
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(R, open(a.out, 'w'), indent=1); print('saved', a.out)
