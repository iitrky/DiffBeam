"""System and algorithm parameters (Table II of the paper)."""
import numpy as np

FC = 100e9                     # carrier frequency (Hz)
C = 3e8
LAM = C / FC                   # wavelength (m)
N = 256                        # BS antennas (ULA)
D_EL = LAM / 2                 # element spacing
DELTA = (np.arange(N) - (N - 1) / 2) * D_EL   # element coordinates (m)
APERTURE = N * D_EL
R_RAYLEIGH = 2 * APERTURE ** 2 / LAM          # ~98.3 m

U_MAX = np.sin(np.deg2rad(60))  # angular sector +/-60 deg (u = sin theta)
R_MIN, R_MAX = 4.0, 30.0        # user distance range (m)
V_MIN, V_MAX = 1 / R_MAX, 1 / R_MIN   # inverse distance range
L_NLOS = 3                      # NLoS scatterers
R_NLOS_MAX = 60.0

M_SENSE = 56                    # Stage-1 sensing beams
R_DEF = 12.0                    # defocus distance of diverging sensing beams (m)
K_HYP = 8                       # Stage-2 hypotheses
PALW_WIN = 10                   # half window (21 beams)
PALW_SCALE = 3.0                # offset normalisation (x grid spacing)

T_DIFF = 50                     # diffusion steps
HIDDEN = 256                    # MLP width

S_POLAR = 8                     # distance rings, exhaustive polar codebook
S_HIER = 16                     # distance candidates, two-stage hierarchical
