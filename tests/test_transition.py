import torch

from yiwm.synth import make_synth_batch
from yiwm.train import train
from yiwm.transition import LearnedTransition, rollout_learned, train_transition


def test_learned_transition_shape_and_bounds():
    tr = LearnedTransition()
    y = torch.rand(8, 6) * 2 - 1
    out = tr(y, torch.randint(0, 5, (8,)))
    assert out.shape == (8, 6) and out.abs().max() <= 1.0


def test_distils_and_rolls_out(tmp_path):
    ckpt = str(tmp_path / "m.pt")
    m = train(steps=300, data="synth", ckpt=ckpt, log_every=999)
    tr, mse = train_transition(m, make_synth_batch, steps=400, pool=2048)
    assert mse < 0.2                                   # fits the one-step map

    r = rollout_learned(m, tr, torch.tanh(torch.randn(6)), steps=25)
    assert 1 <= len(r) <= 25
    assert r[-1]["stop"] in {"maxsteps"} or r[-1]["stop"].startswith("cycle")
    for s in r:
        assert 0 <= s["hex_k"] < 64
    if "cycle_members" in r[-1]:
        assert r[-1]["cycle_len"] == len(r[-1]["cycle_members"])
