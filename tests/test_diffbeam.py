"""Unit tests: run with  python -m pytest -q  (or python tests/test_diffbeam.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffbeam import config, channel, codebooks, conditioning, schemes
from diffbeam.model import DiffBeamModel
from diffbeam.nn import MLP

CKPT = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'diffbeam_defocus_palw.npz')


def test_steering_unit_norm_and_focus():
    b = channel.steer(np.array([0.3]), np.array([10.0]))
    assert np.isclose(np.linalg.norm(b), 1.0)
    assert np.isclose(channel.true_gain(b, b)[0], 1.0)
    assert channel.true_gain(channel.steer(np.array([0.3]), np.array([5.0])), b)[0] < 0.9   # distance matters


def test_rayleigh_distance():
    assert 95 < config.R_RAYLEIGH < 100


def test_channel_power():
    h, u, v = channel.gen_channels(2000, np.random.default_rng(0))
    assert abs(np.mean(np.sum(np.abs(h) ** 2, 1)) / config.N - 1.1) < 0.1


def test_palw_shapes_and_roundtrip():
    rng = np.random.default_rng(1); h, u, v = channel.gen_channels(64, rng)
    W = codebooks.sensing_codebook(56); p = channel.probe(W, h, 100.0, rng)
    cnd = conditioning.Conditioner('palw'); c, ctx = cnd(p)
    assert c.shape == (64, 2 * config.PALW_WIN + 3)
    x = cnd.to_x(u, v, ctx); uu, rr = cnd.from_x(x, ctx)
    ok = np.abs(x[:, 0]) < 1
    assert np.allclose(uu[ok], u[ok]) and np.allclose(1 / rr, v)


def test_mlp_gradient():
    rng = np.random.default_rng(0); net = MLP([3, 8, 2], rng); x = rng.standard_normal((5, 3)); y = rng.standard_normal((5, 2))
    W0 = net.W[0].copy(); out = net.forward(x, cache=True); loss0 = np.mean((out - y) ** 2)
    net.step(2 * (out - y) / out.size, 1e-3)
    assert np.mean((net.forward(x) - y) ** 2) < loss0 and not np.allclose(W0, net.W[0])


def test_checkpoint_beats_far_field_at_0dB():
    rng = np.random.default_rng(3); m = DiffBeamModel.load(CKPT)
    h, _, _ = channel.gen_channels(200, rng)
    r_db = np.mean(schemes.diffbeam(h, 1.0, rng, m, K=8))
    r_ff = np.mean(schemes.far_field_dft(h, 1.0, rng))
    assert r_db > r_ff + 0.5, (r_db, r_ff)


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'): f(); print('PASS', k)
