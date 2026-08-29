import numpy as np
import pandas as pd
import pytest

from yiwm.market_adapter import load_market, market_to_yao


@pytest.fixture
def fake_csv(tmp_path):
    n = 400
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({"Close": close, "Volume": rng.integers(1e6, 5e6, n),
                       "High": close * 1.01, "Low": close * 0.99, "Open": close}, index=idx)
    p = str(tmp_path / "x.csv")
    df.to_csv(p)
    return p


def test_load_and_map_no_lookahead(fake_csv):
    f = load_market(fake_csv)
    assert {"mom_5", "mom_20", "vol", "vol_flow", "trend", "revert", "fwd_ret_5"} <= set(f.columns)
    assert f.isna().sum().sum() == 0
    y = market_to_yao(f)
    assert y.shape == (len(f), 6)
    assert np.abs(y).max() <= 1.0

    # fwd_ret_5 is a forward label; the 6 input features must not depend on it
    f2 = f.copy()
    f2.loc[f2.index[-10:], "fwd_ret_5"] = 999.0            # scramble the future
    assert np.array_equal(market_to_yao(f2), y)            # obs unchanged


def test_market_regime_is_deterministic_label(fake_csv):
    from yiwm.market_adapter import market_regime

    f = load_market(fake_csv)
    r = market_regime(f)
    assert list(r.columns) == ["hex", "king_wen", "yang_dims"]
    assert len(r) == len(f)
    assert r["king_wen"].between(1, 64).all()
    assert (r["yang_dims"].str.len() == 6).all()
    # deterministic + no forward leakage
    f2 = f.copy()
    f2.loc[f2.index[-5:], "fwd_ret_5"] = -5.0
    assert list(market_regime(f2)["king_wen"]) == list(r["king_wen"])


def test_market_transition_trains(fake_csv):
    from yiwm.market_transition import MarketTransition, _splits
    import torch
    import torch.nn.functional as F

    y = market_to_yao(load_market(fake_csv))
    xtr, ttr, xte, tte = _splits(y)
    m = MarketTransition()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    l0 = F.mse_loss(m(xte), tte).item()
    for _ in range(120):
        loss = F.mse_loss(m(xtr), ttr)
        opt.zero_grad(); loss.backward(); opt.step()
    l1 = F.mse_loss(m(xte), tte).item()
    assert l1 < l0                                          # learns something on held-out
