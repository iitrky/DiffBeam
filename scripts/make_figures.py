"""Regenerate paper figures from results/*.json  ->  figures/"""
import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffbeam.channel import steer
from diffbeam.codebooks import sensing_codebook
os.makedirs('figures', exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3})
R = json.load(open('results/results_main.json'))
ST = {'Perfect CSI': ('k', '-', None), 'Exhaustive polar': ('tab:gray', '--', 's'),
      'Two-stage hierarchical': ('tab:green', '-.', '^'), 'Far-field DFT': ('tab:purple', ':', 'v'),
      'DNN regression': ('tab:orange', '--', 'x'), 'DNN + refinement': ('tab:red', '-', 'd'),
      'DiffBeam (K=1)': ('tab:cyan', '--', 'o'), 'DiffBeam': ('tab:blue', '-', 'o')}
OV = {'Exhaustive polar': 2048, 'Two-stage hierarchical': 272, 'Far-field DFT': 256,
      'DNN regression': 56, 'DNN + refinement': 64, 'DiffBeam (K=1)': 56, 'DiffBeam': 64}
def lab(k): return k + (f' [{OV[k]}]' if k in OV else '')
# Fig 2: rate vs SNR
plt.figure(figsize=(5, 3.8))
for k, v in R['vs_snr'].items():
    col, ls, mk = ST[k]; plt.plot(R['snr'], v, color=col, ls=ls, marker=mk, ms=5, label=lab(k))
plt.xlabel('Pilot/data SNR per antenna (dB)'); plt.ylabel('Achievable rate (bit/s/Hz)')
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig('figures/fig_rate_snr.png', dpi=200); plt.close()
# Fig 3: rate vs K
plt.figure(figsize=(5, 3.6))
for s, mk in [(0, 'o'), (-10, 's')]:
    plt.semilogx(R['Ks'], R['vs_K'][f'DiffBeam@{s}'], 'b-', marker=mk, base=2, label=f'DiffBeam, {s} dB')
    plt.semilogx(R['Ks'], R['vs_K'][f'DNN + refinement@{s}'], 'r--', marker=mk, base=2, label=f'DNN + refinement, {s} dB')
    plt.semilogx(R['Ks'], R['vs_K'][f'Perfect CSI@{s}'], 'k:', base=2, label=f'Perfect CSI, {s} dB')
plt.xlabel('Number of refinement candidates K'); plt.ylabel('Achievable rate (bit/s/Hz)')
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig('figures/fig_rate_K.png', dpi=200); plt.close()
# Fig 4: rate vs distance
plt.figure(figsize=(5, 3.6)); xl = [f'{a}-{b}' for a, b in R['dbins']]
for k in ['Perfect CSI', 'Exhaustive polar', 'Two-stage hierarchical', 'Far-field DFT', 'DNN + refinement', 'DiffBeam']:
    col, ls, mk = ST[k]; plt.plot(xl, R['vs_dist'][k], color=col, ls=ls, marker=mk, label=k)
plt.xlabel('User distance range (m)'); plt.ylabel('Achievable rate (bit/s/Hz)')
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig('figures/fig_rate_dist.png', dpi=200); plt.close()
# Fig 1: sensing beam patterns (system illustration)
u = np.linspace(-1, 1, 2000)
fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
for kind, a in [('dft', ax[0]), ('defocus', ax[1])]:
    W = sensing_codebook(56, kind)
    for r in [5, 25]:
        G = np.abs(steer(u, r * np.ones_like(u)).conj() @ W.T) ** 2
        a.plot(u, 10 * np.log10(G.max(1) + 1e-6), label=f'r = {r} m')
    a.set_ylim(-35, 2); a.set_xlabel('sin(θ)'); a.set_ylabel('Best-beam gain (dB)')
    a.set_title('Far-field DFT sensing' if kind == 'dft' else 'Proposed defocused sensing', fontsize=9); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig('figures/fig_sensing.png', dpi=200); plt.close()
print('figs ok')

# ---- Fig. 1: block diagram
from matplotlib.patches import FancyBboxPatch

fig,ax=plt.subplots(figsize=(7.2,2.2)); ax.axis('off'); ax.set_xlim(0,10.4); ax.set_ylim(0,2.2)
B=[("Stage 1\nDefocused sensing\nsweep (M beams)","#dbe9f6"),("PALW\nconditioning\n(peak + window)","#e8f3e0"),
   ("Conditional DDPM\nposterior sampler\n(K candidates)","#fde9d9"),("Stage 2\nCandidate probing\n(K pencil beams)","#dbe9f6"),("Selected near-field\nbeam  b(θ̂, r̂)","#eeeeee")]
for i,(t,col) in enumerate(B):
    x=0.1+i*2.08
    ax.add_patch(FancyBboxPatch((x,0.55),1.75,1.1,boxstyle="round,pad=0.05",fc=col,ec='k',lw=0.8))
    ax.text(x+0.875,1.1,t,ha='center',va='center',fontsize=8)
    if i<4: ax.annotate('',xy=(x+2.05,1.1),xytext=(x+1.82,1.1),arrowprops=dict(arrowstyle='->',lw=1))
ax.text(0.975,0.3,'p ∈ ℝ^M (received powers)',ha='center',fontsize=7)
ax.text(3.055,0.3,'c = [window, u_pk, log p_max]',ha='center',fontsize=7)
ax.text(5.135,0.3,'x̂^(k) = (Δû, v̂), k=1..K',ha='center',fontsize=7)
ax.text(7.215,0.3,'k* = argmax_k p̃_k',ha='center',fontsize=7)
ax.text(5.2,2.0,'Total pilot overhead  T = M + K  (= 56 + 8 = 64)',ha='center',fontsize=8,style='italic')
plt.tight_layout(); plt.savefig('figures/fig_block.png',dpi=220)
