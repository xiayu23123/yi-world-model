import torch

from yiwm.encoders import GridEncoder
from yiwm.kingdom_2d import Kingdom2D
from yiwm.kingdom_stage2 import eval_vs_random, train_dqn


def test_grid_encoder_two_heads():
    enc = GridEncoder(size=5)
    e = Kingdom2D(size=5, seed=0)
    out = enc(e.grid(), e.player())
    assert out["yao"].shape == (1, 6) and out["dyn"].shape == (1, 32)
    assert out["yao"].abs().max() <= 1.0
    b = enc(torch.rand(4, 3, 5, 5), torch.rand(4, 3))
    assert b["yao"].shape == (4, 6) and b["dyn"].shape == (4, 32)


def test_grid_dqn_beats_random_and_dyn_not_collapsed():
    enc, q = train_dqn(size=5, episodes=1200, seed=0)
    g, r = eval_vs_random(enc, q, size=5, n=60)
    assert g > r + 5.0

    ds = []
    for s in range(120):
        e = Kingdom2D(size=5, seed=s + 500)
        for _ in range(s % 8):
            e.step(s % 5)
        with torch.no_grad():
            ds.append(enc(e.grid(), e.player())["dyn"][0])
    assert torch.stack(ds).std(0).mean() > 0.02        # LayerNorm dyn is not a constant
