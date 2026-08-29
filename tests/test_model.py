import torch

from yiwm.data import OBS_DIM, make_batch
from yiwm.losses import yi_world_loss
from yiwm.model import YiWorldModel


def _batch(n=8):
    return make_batch(n, seed=123)


def test_forward_shapes():
    m = YiWorldModel(obs_dim=OBS_DIM)
    b = _batch()
    o = m(b["obs"], b["entity_states"], b["entity_cats"], b["entity_adj"])
    assert o["hex_logits"].shape == (8, 64)
    assert o["hex_logits_next"].shape == (8, 64)
    assert o["change"].shape == (8, 6)
    assert o["policy"]["action_logits"].shape == (8, 5)
    assert o["policy"]["intensity"].shape == (8, 1)
    assert o["policy"]["yao_advice"].shape == (8, 6)
    assert o["entity_next"].shape == (8, 5, 6)


def test_backward_produces_grads():
    m = YiWorldModel(obs_dim=OBS_DIM)
    b = _batch()
    o = m(b["obs"], b["entity_states"], b["entity_cats"], b["entity_adj"])
    L = yi_world_loss(o, b)
    L["total"].backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    # gradient must reach HexagramInference through the ChangeEngine path too
    assert m.hexinf.hex_embed.weight.grad.abs().sum() > 0
    # multi-relation graph conv: gates and every relation weight get gradient
    assert m.hexinf.rel_gate.grad.abs().sum() > 0
    for layer in m.hexinf.rel_lin:
        for r, lin in layer.items():
            assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0, r


def test_relation_weights_shape():
    m = YiWorldModel(obs_dim=OBS_DIM)
    g = m.hexinf.relation_weights()
    assert g.shape == (2, len(m.hexinf.RELATIONS))
    assert torch.allclose(g.sum(1), torch.ones(2), atol=1e-5)


def test_adaptive_threshold_is_input_dependent():
    m = YiWorldModel(obs_dim=OBS_DIM)
    ce = m.change
    y1 = torch.tensor([[0.9, 0.1, 0.1, 0.1, 0.1, 0.1]])
    y2 = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 0.9]])
    t1 = ce.adaptive_threshold(y1)
    t2 = ce.adaptive_threshold(y2)
    assert t1.shape == (1, 6) and (t1 > 0).all()
    # same |y| at position 0 (0.9) but different rank context -> different cutoff
    assert not torch.allclose(t1[0, 0], t2[0, 5], atol=1e-4) or not torch.allclose(t1, t2)


def test_joint_moving_head():
    from yiwm.constants import MOVING_MASKS
    m = YiWorldModel(obs_dim=OBS_DIM)
    b = _batch()
    o = m(b["obs"], b["entity_states"], b["entity_cats"], b["entity_adj"])
    assert o["moving_logits"].shape == (8, 21)
    assert o["hex_logits_next_joint"].shape == (8, 64)
    # mask-index helper is the inverse of the mask table
    idx = m.moving_mask_index(MOVING_MASKS)
    assert torch.equal(idx, torch.arange(21))
    # a 3-line mask is not in the 1/2-line vocabulary -> -1
    bad = torch.tensor([[1.0, 1, 1, 0, 0, 0]])
    assert m.moving_mask_index(bad).item() == -1
    # loss dict carries the joint terms
    from yiwm.losses import yi_world_loss
    L = yi_world_loss(o, b)
    assert "moving_joint" in L and "hex_next_joint" in L


def test_change_energy_and_rank_loss():
    from yiwm.losses import _moving_rank_loss
    m = YiWorldModel(obs_dim=OBS_DIM)
    b = _batch()
    o = m(b["obs"], b["entity_states"], b["entity_cats"], b["entity_adj"])
    assert o["change_energy"].shape == (8, 6)
    # perfectly separated energy -> zero ranking loss; reversed -> positive
    moving = torch.tensor([[1.0, 1, 0, 0, 0, 0]])
    good = torch.tensor([[5.0, 5, -5, -5, -5, -5]])
    bad = torch.tensor([[-5.0, -5, 5, 5, 5, 5]])
    assert _moving_rank_loss(good, moving, 0.3).item() == 0.0
    assert _moving_rank_loss(bad, moving, 0.3).item() > 0.0


def test_soft_hex_target_is_distribution():
    from yiwm.constants import BINARY_HEX
    from yiwm.losses import _soft_hex_target
    y = torch.randn(32, 6)
    t = _soft_hex_target(y, temp=0.15)
    assert t.shape == (32, 64)
    assert torch.allclose(t.sum(1), torch.ones(32), atol=1e-4)
    # peak of the soft target = the hard-sign hexagram
    hard = ((y > 0).long() * (2 ** torch.arange(6))).sum(1)
    assert torch.equal(t.argmax(1), hard)


def test_change_engine_hard_is_binary():
    m = YiWorldModel(obs_dim=OBS_DIM)
    m.eval()
    b = _batch()
    with torch.no_grad():
        o = m(b["obs"], b["entity_states"], b["entity_cats"], b["entity_adj"], hard=True)
    uniq = set(o["change"].unique().tolist())
    assert uniq.issubset({0.0, 1.0})


def test_king_wen_roundtrip_on_model():
    m = YiWorldModel(obs_dim=OBS_DIM)
    k = torch.arange(64)
    assert torch.equal(m.from_king_wen(m.to_king_wen(k)), k)
