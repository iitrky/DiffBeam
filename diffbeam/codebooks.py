"""Sensing, far-field DFT and polar-domain codebooks."""
import numpy as np
from .config import N, U_MAX, V_MIN, V_MAX, R_DEF
from .channel import steer, steer_fresnel


def sensing_codebook(M, kind='defocus', r_def=R_DEF):
    """Proposed diverging defocused sensing beams (kind='defocus') or a far-field DFT subset (kind='dft')."""
    u = np.linspace(-U_MAX, U_MAX, M)
    curv = -1.0 / r_def if kind == 'defocus' else 0.0
    return steer_fresnel(u, curv * np.ones(M))


def dft_grid():
    ug = np.linspace(-1, 1, N, endpoint=False) + 1 / N
    return ug[np.abs(ug) <= U_MAX + 1 / N]


def dft_codebook():
    ug = dft_grid()
    return ug, steer_fresnel(ug, np.zeros(len(ug)))


def polar_codebook(S):
    """Angle grid x S distance rings uniform in 1/r (polar-domain codebook)."""
    ug = dft_grid(); vr = np.linspace(V_MIN, V_MAX, S)
    return np.concatenate([steer(ug, 1 / v * np.ones(len(ug))) for v in vr])
