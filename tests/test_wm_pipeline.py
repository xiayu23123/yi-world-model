from yiwm.wm_pipeline import QLatent, eval_policies, train_q_frozen
from yiwm.wm_pretrain import pretrain, report


def test_full_pipeline_frozen_latent_supports_policy():
    enc, tr, _ = pretrain(size=5, n_ep=500, epochs=20, seed=0)
    _, sens_before = report(enc, tr, size=5, n=150)

    q = train_q_frozen(enc, size=5, episodes=900, seed=0)

    # encoder + transition are genuinely frozen
    assert all(not p.requires_grad for p in enc.parameters())
    _, sens_after = report(enc, tr, size=5, n=150)
    assert abs(sens_after - sens_before) < 0.05

    res = eval_policies(enc, tr, q, size=5, n=60, horizon=3)
    assert res["greedy"] > res["random"] * 1.15        # Q on the frozen latent learns a policy
    assert isinstance(q, QLatent)
