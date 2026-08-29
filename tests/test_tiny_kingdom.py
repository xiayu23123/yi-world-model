import torch

from yiwm.tiny_kingdom import (
    TinyKingdom, action_sensitivity, collect, train,
)


def test_env_step_and_causality():
    e = TinyKingdom(seed=1)
    s = (e.resource, e.morale, e.threat)
    o, r, done, info = e.step(0)
    assert o.shape == (6,) and (o >= 0).all() and (o <= 1).all()
    assert isinstance(r, float) and set(info) == {"resource", "morale", "threat"}

    # same state, different action -> different next obs (action is causal)
    e.resource, e.morale, e.threat, e.t = *s, 0
    a = e.step(0)[0]
    e.resource, e.morale, e.threat, e.t = *s, 0
    b = e.step(2)[0]
    assert (a - b).abs().mean() > 1e-3


def test_transition_learns_action_dependence():
    buf = collect(n_episodes=400, seed=0)
    assert len(buf) > 3000
    tr, val, mse = train(buf, epochs=25)
    assert mse < 0.01                       # clean env -> near-perfect fit
    assert action_sensitivity(tr, n=150) > 0.03   # learned action-dependent dynamics
