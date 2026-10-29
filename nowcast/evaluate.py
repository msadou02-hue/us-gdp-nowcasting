"""Pseudo out-of-sample evaluation with an expanding window."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nowcast.models import clone_model

PERIODS = {
    "Pre-COVID (2012-2019)": ("2012-01-01", "2019-12-31"),
    "COVID (2020-2021)": ("2020-01-01", "2021-12-31"),
    "Post-COVID (2022-)": ("2022-01-01", None),
    "Full test sample": (None, None),
}


def add_lag(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    out = panel.copy()
    out["y_lag"] = out[target].shift(1)
    return out.iloc[1:]


def expanding_window(panel: pd.DataFrame, target: str, models: dict,
                     test_start: str = "2012-01-01", min_coverage: float = 0.8,
                     verbose: bool = True) -> pd.DataFrame:
    """Nowcast each quarter from `test_start` using only data from earlier quarters.

    At quarter t the model is trained on quarters < t and predicts y_t from the
    indicators observed during quarter t. Indicators with less than
    `min_coverage` non-missing history in the training window are dropped.
    """
    data = add_lag(panel, target)
    test_dates = data.index[data.index >= test_start]
    preds = {name: [] for name in models}
    for i, t in enumerate(test_dates):
        train = data.loc[data.index < t]
        cols = [c for c in train.columns if c != target and train[c].notna().mean() >= min_coverage]
        X_tr, y_tr = train[cols], train[target]
        X_te = data.loc[[t], cols]
        for name, model in models.items():
            m = clone_model(model).fit(X_tr, y_tr)
            preds[name].append(float(m.predict(X_te)[0]))
        if verbose and (i % 8 == 0 or i == len(test_dates) - 1):
            print(f"  {t.year}Q{t.quarter}  ({i + 1}/{len(test_dates)})")
    out = pd.DataFrame(preds, index=test_dates)
    out.insert(0, "Actual", data.loc[test_dates, target])
    out.insert(1, "y_lag", data.loc[test_dates, "y_lag"])
    return out


def metrics(actual: pd.Series, pred: pd.Series, lag: pd.Series) -> dict:
    err = pred - actual
    # Directional accuracy: did the model get the sign of the change vs last quarter right?
    mda = np.mean(np.sign(actual - lag) == np.sign(pred - lag))
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err))),
            "MDA": float(mda)}


def score_table(nowcasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_cols = [c for c in nowcasts.columns if c not in ("Actual", "y_lag")]
    for period, (lo, hi) in PERIODS.items():
        sub = nowcasts.loc[lo:hi]
        if sub.empty:
            continue
        ar_rmse = metrics(sub["Actual"], sub["AR(1)"], sub["y_lag"])["RMSE"]
        for name in model_cols:
            m = metrics(sub["Actual"], sub[name], sub["y_lag"])
            rows.append({"Period": period, "Model": name, **m,
                         "RMSE vs AR(1)": m["RMSE"] / ar_rmse, "N": len(sub)})
    return pd.DataFrame(rows)
