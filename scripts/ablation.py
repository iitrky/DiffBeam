"""Ablation study (Table IV): full DiffBeam vs. w/o multi-hypothesis, w/o PALW, w/o defocused sensing."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffbeam.model import DiffBeamModel
from diffbeam.channel import gen_channels
from diffbeam import schemes

ap = argparse.ArgumentParser()
ap.add_argument('--full', default='checkpoints/diffbeam_defocus_palw.npz')
ap.add_argument('--no_palw', default='checkpoints/diffbeam_defocus_global.npz')
ap.add_argument('--no_defocus', default='checkpoints/diffbeam_dft_palw.npz')
ap.add_argument('--B', type=int, default=800); ap.add_argument('--K', type=int, default=8)
ap.add_argument('--out', default='results/results_ablation.json')
a = ap.parse_args()
rng = np.random.default_rng(77)
full, nopalw, nodef = (DiffBeamModel.load(p) for p in (a.full, a.no_palw, a.no_defocus))
out = {}
for s in [-5, 0, 10]:
    snr = 10 ** (s / 10); h, _, _ = gen_channels(a.B, rng)
    r = {'Full DiffBeam': schemes.diffbeam(h, snr, rng, full, K=a.K),
         'w/o multi-hypothesis (K=1)': schemes.diffbeam(h, snr, rng, full, K=1),
         'w/o PALW conditioning': schemes.diffbeam(h, snr, rng, nopalw, K=a.K),
         'w/o defocused sensing (DFT subset)': schemes.diffbeam(h, snr, rng, nodef, K=a.K)}
    out[s] = {k: float(np.mean(v)) for k, v in r.items()}
    print(f'{s:3d} dB', {k: round(v, 2) for k, v in out[s].items()}, flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(out, open(a.out, 'w'), indent=1); print('saved', a.out)
