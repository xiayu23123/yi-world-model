"""Smoke test: the toy target is learnable -- loss drops, 本卦 acc climbs."""

import torch

from yiwm.data import OBS_DIM, make_batch
from yiwm.losses import yi_world_loss
from yiwm.model import YiWorldModel


def test_dataset_is_self_consistent():
    b = make_batch(64, seed=1)
    # moving lines flip 本卦 into 之卦 exactly
    from yiwm.constants import BINARY_HEX
    cur = BINARY_HEX[b["hex"]]
    nxt = BINARY_HEX[b["hex_next"]]
    assert torch.equal((cur != nxt).long(), b["moving"])


def test_overfits_small_sample():
    torch.manual_seed(0)
    model = YiWorldModel(obs_dim=OBS_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    fixed = make_batch(512, seed=42)

    def acc():
        with torch.no_grad():
            o = model(fixed["obs"], fixed["entity_states"],
                      fixed["entity_cats"], fixed["entity_adj"])
            return (o["hex_logits"].argmax(1) == fixed["hex"]).float().mean().item()

    start = acc()
    model.train()
    for _ in range(300):
        o = model(fixed["obs"], fixed["entity_states"],
                  fixed["entity_cats"], fixed["entity_adj"])
        L = yi_world_loss(o, fixed)
        opt.zero_grad()
        L["total"].backward()
        opt.step()
    end = acc()
    assert end > start + 0.3
    assert end > 0.8
