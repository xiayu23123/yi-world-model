from yiwm.wm_pretrain import pretrain, report


def test_dynamics_first_latent_recovers_action_sensitivity():
    enc, tr, loss = pretrain(size=5, n_ep=500, epochs=25, seed=0)
    across, sens = report(enc, tr, size=5, n=200)
    assert loss < 0.1
    assert across > 0.3                 # variance reg -> latent not collapsed
    # dynamics-first latent: action-conditioned transition is learnable again
    # (stage 1 hand-R^6 = 0.026, stage 2 DQN-latent = 0.014)
    assert sens > 0.05
