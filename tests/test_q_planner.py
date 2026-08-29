import torch

from yiwm.q_planner import QNet, eval_greedy_vs_random, train_q


def test_qnet_shapes():
    q = QNet()
    s = torch.rand(4, 6)
    assert q(s, torch.tensor([0, 1, 2, 3])).shape == (4,)
    assert q.all_q(s).shape == (4, 5)
    assert q.all_q(torch.rand(6)).shape == (1, 5)


def test_q_greedy_beats_random():
    q = train_q(episodes=1200, seed=0)
    g, r = eval_greedy_vs_random(q, n=120)
    assert g > r + 1.0            # action-aware Q -> a real planner (naive V-MPC lost)
