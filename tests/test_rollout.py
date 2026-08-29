import torch

from yiwm.model import YiWorldModel


def test_rollout_shapes_and_termination():
    m = YiWorldModel(obs_dim=35)
    m.eval()
    r = m.rollout(torch.tanh(torch.randn(6)), steps=6)
    assert 1 <= len(r) <= 6
    for s in r:
        assert 0 <= s["hex_k"] < 64 and 0 <= s["hex_next_k"] < 64
        assert len(s["moving"]) == 6 and sum(s["moving"]) in (0, 1, 2)
    assert r[-1].get("stop") in {"fixed", None} or r[-1]["stop"].startswith("cycle")


def test_rollout_decays_and_reaches_equilibrium():
    m = YiWorldModel(obs_dim=35)
    m.eval()
    r = m.rollout(torch.tanh(torch.randn(6) * 0.5), steps=30, decay=0.6)
    mags = [s["mag"] for s in r]
    assert mags[-1] <= mags[0] + 1e-6              # magnitude never grows
    # strong decay + long horizon -> must terminate (fixed or cycle), not maxsteps
    assert r[-1].get("stop") is not None


def test_rollout_flip_is_consistent():
    m = YiWorldModel(obs_dim=35)
    m.eval()
    r = m.rollout(torch.tanh(torch.randn(6)), steps=8)
    for s in r:
        flip = sum(b << i for i, b in enumerate(s["moving"]))
        assert s["hex_next_k"] == s["hex_k"] ^ flip
