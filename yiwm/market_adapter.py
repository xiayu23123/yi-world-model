"""Map a market price series to the R^6 yao-force space — a real external
environment for `transition.py` (the P3 limit was "needs a real env").

Deterministic, no learning, **no look-ahead**: every feature is backward-looking
so `yao_t` is knowable at time t. The 6 dims:

  0 mom_5    5-day return              阳 = up
  1 mom_20   20-day return             阳 = up
  2 vol      20-day realised vol       |·| large = 老 (volatile)
  3 vol_flow 20-day volume ratio       阳 = expanding volume
  4 trend    price vs 60-day MA        global-anchor dim (五爻-ish)
  5 revert   price vs 20-day MA        阴 = stretched above (mean-revert down)

This is a research mapping for methodology testing, not a trading signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_market(source: str, start="2015-01-01", end="2025-01-01") -> pd.DataFrame:
    """source = a CSV path or a ticker. Returns a frame with the 6 raw features
    (backward-looking) + `fwd_ret_5` (a *label*, forward 5-day return)."""
    if source.endswith(".csv"):
        px = pd.read_csv(source, index_col=0, parse_dates=True)
    else:
        import yfinance as yf

        px = yf.download(source, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(px.columns, pd.MultiIndex):
            px.columns = px.columns.get_level_values(0)
    close, vol = px["Close"].astype(float), px["Volume"].astype(float)
    ret = close.pct_change()

    f = pd.DataFrame(index=px.index)
    f["mom_5"] = close.pct_change(5)
    f["mom_20"] = close.pct_change(20)
    f["vol"] = ret.rolling(20).std()
    f["vol_flow"] = vol / vol.rolling(20).mean() - 1.0
    f["trend"] = close / close.rolling(60).mean() - 1.0
    f["revert"] = close / close.rolling(20).mean() - 1.0
    f["fwd_ret_5"] = close.pct_change(5).shift(-5)          # LABEL, not an input
    return f.dropna()


_SCALE = np.array([6.0, 4.0, 25.0, 1.5, 5.0, 8.0])          # feature -> ~[-1,1]
_CENTER = np.array([0.0, 0.0, 0.35, 0.0, 0.0, 0.0])         # vol is a magnitude, recentre


def market_to_yao(df: pd.DataFrame) -> np.ndarray:
    """[N, 6] signed force vectors from the 6 raw feature columns."""
    x = df[["mom_5", "mom_20", "vol", "vol_flow", "trend", "revert"]].to_numpy(float)
    y = np.tanh((x - _CENTER) * _SCALE)
    y[:, 2] = np.tanh((x[:, 2] - _CENTER[2]) * _SCALE[2])   # vol: sign = above/below normal
    return y


def market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic market-state -> 卦 labelling (NOT a prediction).

    `sign(yao)` of the 6 indicators is read as a 6-bit hexagram: a structured
    name for 'what the tape looks like right now', nothing more. No learning,
    no forward claim.
    """
    from .constants import BINARY_TO_KING_WEN, KING_WEN_NAMES

    y = market_to_yao(df)
    bits = (y > 0).astype(int)
    k = bits @ (2 ** np.arange(6))
    kw = BINARY_TO_KING_WEN.numpy()[k]
    return pd.DataFrame({
        "hex": [KING_WEN_NAMES[j] for j in kw],
        "king_wen": kw + 1,
        "yang_dims": ["".join("阳" if b else "阴" for b in row) for row in bits],
    }, index=df.index)
