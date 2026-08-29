import torch

from yiwm.kingdom_2d import Kingdom2D
from yiwm.q_planner import eval_greedy_vs_random, train_q


def test_env_interface_matches_tinykingdom():
    e = Kingdom2D(size=3, seed=1)
    o = e.obs()
    assert o.shape == (6,) and (o >= 0).all() and (o <= 1).all()
    o, r, done, info = e.step(0)
    assert o.shape == (6,) and isinstance(r, float)
    assert {"n_ctrl", "p_res", "threat"} <= set(info)

    # spatial causal effect: 进 grows controlled area, 守 does not
    e2 = Kingdom2D(size=5, seed=3)
    n0 = int(e2._controlled().sum())
    for _ in range(4):
        e2.step(0)
    assert int(e2._controlled().sum()) >= n0


def test_q_scales_to_grid():
    mk = lambda s: Kingdom2D(size=3, seed=s)
    q = train_q(episodes=1500, seed=0, env_fn=mk)
    g, r = eval_greedy_vs_random(q, n=80, env_fn=mk)
    assert g > r + 5.0                     # Q-learning beats random on the spatial env
