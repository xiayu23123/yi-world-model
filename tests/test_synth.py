import torch

from yiwm.constants import BINARY_HEX
from yiwm.losses import yi_world_loss
from yiwm.model import YiWorldModel
from yiwm.synth import OBS_DIM_SYNTH, make_synth_batch


def test_shapes_and_ranges():
    b = make_synth_batch(16, seed=0)
    assert b["obs"].shape == (16, OBS_DIM_SYNTH)
    assert b["entity_states"].shape == (16, 5, 6)
    assert b["entity_cats"].shape == (16, 5)
    assert b["entity_adj"].shape == (16, 5, 5)
    assert b["hex"].shape == (16,) and b["moving"].shape == (16, 6)
    assert int(b["entity_cats"].min()) >= 0 and int(b["entity_cats"].max()) <= 4
    assert int(b["action"].min()) >= 0 and int(b["action"].max()) <= 4
    assert len(b["text"]) == 16 and isinstance(b["text"][0], str)


def test_force_sign_matches_benGua():
    """The draft's bug: moving lines had flipped sign, contradicting 本卦."""
    b = make_synth_batch(256, seed=1)
    sign_bits = (b["yao_target"] > 0).long()
    assert torch.equal(sign_bits, BINARY_HEX[b["hex"]].long())


def test_moving_maps_ben_to_zhi():
    b = make_synth_batch(256, seed=2)
    cur = BINARY_HEX[b["hex"]].long()
    nxt = BINARY_HEX[b["hex_next"]].long()
    assert torch.equal((cur != nxt).long(), b["moving"].long())
    assert b["moving"].sum(1).min() >= 1 and b["moving"].sum(1).max() <= 2


def test_moving_lines_are_extreme():
    b = make_synth_batch(256, seed=3)
    mv = b["moving"].bool()
    assert b["yao_target"].abs()[mv].min() > 0.85
    assert b["yao_target"].abs()[~mv].max() < 0.85


def test_labels_not_degenerate():
    b = make_synth_batch(512, seed=4)
    assert b["action"].unique().numel() >= 3
    assert b["timing"].unique().numel() >= 4


def test_deterministic_by_seed():
    a = make_synth_batch(32, seed=7)
    c = make_synth_batch(32, seed=7)
    for k in ("hex", "hex_next", "action", "timing"):
        assert torch.equal(a[k], c[k])
    assert a["text"] == c["text"]


def test_learnable():
    torch.manual_seed(0)
    model = YiWorldModel(obs_dim=OBS_DIM_SYNTH)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    fixed = make_synth_batch(512, seed=42)

    def acc(key_logits, key_tgt):
        with torch.no_grad():
            o = model(fixed["obs"], fixed["entity_states"],
                      fixed["entity_cats"], fixed["entity_adj"])
        return (o[key_logits].argmax(1) == fixed[key_tgt]).float().mean().item()

    h0, a0 = acc("hex_logits", "hex"), None
    model.train()
    for _ in range(300):
        o = model(fixed["obs"], fixed["entity_states"],
                  fixed["entity_cats"], fixed["entity_adj"])
        L = yi_world_loss(o, fixed)
        opt.zero_grad()
        L["total"].backward()
        opt.step()
    with torch.no_grad():
        o = model(fixed["obs"], fixed["entity_states"],
                  fixed["entity_cats"], fixed["entity_adj"])
    h1 = (o["hex_logits"].argmax(1) == fixed["hex"]).float().mean().item()
    act1 = (o["policy"]["action_logits"].argmax(1) == fixed["action"]).float().mean().item()
    assert h1 > h0 + 0.3 and h1 > 0.7
    assert act1 > 0.6          # action is learnable from structure (draft: it was noise)
