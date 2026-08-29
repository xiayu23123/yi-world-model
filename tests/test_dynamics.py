import json

from yiwm.dynamics import analyse
from yiwm.train import train


def test_dynamics_portrait(tmp_path):
    ckpt = str(tmp_path / "m.pt")
    train(steps=200, data="synth", ckpt=ckpt, log_every=999)   # cheap real ckpt
    stats = analyse(ckpt, steps=12, trials=3)

    assert len(stats) == 64
    for h, s in stats.items():
        assert 1.0 <= s["mean_steps"] <= 12.0
        assert 0.0 <= s["cycle_frac"] <= 1.0 and 0.0 <= s["fixed_frac"] <= 1.0
        assert 0.0 < s["mean_decay"] <= 1.001
        assert isinstance(s["self_attractor"], bool)

    assert (tmp_path / "rollout_stats.json").exists()
    pc = (tmp_path / "perturbation.csv").read_text(encoding="utf-8").splitlines()
    assert pc[0] == "hex,yao,sigma,flip_benGua,flip_zhiGua,flip_action"
    assert len(pc) == 1 + 64 * 6 * 3                            # 384 rows x 3 sigmas
    # round-trip the json
    j = json.loads((tmp_path / "rollout_stats.json").read_text(encoding="utf-8"))
    assert set(j) == set(stats)
