#!/usr/bin/env bash
# Reproduce every result in the paper from scratch (~25 min on one CPU core).
set -e
python scripts/train.py --sensing defocus --conditioning palw   --iters 15000 --out checkpoints/diffbeam_defocus_palw.npz
python scripts/train.py --sensing dft     --conditioning palw   --iters 15000 --out checkpoints/diffbeam_dft_palw.npz
python scripts/train.py --sensing defocus --conditioning global --iters 12000 --out checkpoints/diffbeam_defocus_global.npz
python scripts/evaluate.py
python scripts/ablation.py
python scripts/make_figures.py
