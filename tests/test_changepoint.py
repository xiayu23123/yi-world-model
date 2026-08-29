import numpy as np

from yiwm.changepoint import (
    binseg_l2, cusum_raw, feature_cusum, pelt_l2, score, synth, window_to_yao,
)


def test_window_to_yao_bounded():
    y = window_to_yao(np.random.default_rng(0).normal(3, 2, 50))
    assert y.shape == (6,) and np.abs(y).max() <= 1.0


def test_baselines_find_a_strong_shift():
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(0, 0.5, 300), rng.normal(6, 0.5, 300)])
    for det in (lambda x: pelt_l2(x, 12),
                lambda x: binseg_l2(x, 1),
                cusum_raw):
        cps = det(s)
        assert any(280 <= c <= 320 for c in cps), det


def test_score_and_synth_shapes():
    s, true = synth(n=800, n_cp=4, seed=1)
    assert len(s) == 800 and len(true) == 4
    r = score([true[0] + 3, 999], true, n=800)
    assert 0.0 <= r["f1"] <= 1.0 and r["fp"] == 1


def test_feature_cusum_runs():
    s, _ = synth(seed=2)
    cps = feature_cusum(s)
    assert isinstance(cps, list) and all(0 <= c <= len(s) for c in cps)
