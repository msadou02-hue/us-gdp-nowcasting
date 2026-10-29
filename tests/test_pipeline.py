import numpy as np
import pandas as pd

from nowcast.data import transform
from nowcast.evaluate import expanding_window, metrics, score_table
from nowcast.models import AR1, FactorModel


def _synthetic_panel(n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-03-31", periods=n, freq="QE-DEC")
    f = rng.normal(size=n)
    data = {f"x{i}": f * rng.uniform(0.5, 1.5) + rng.normal(scale=0.5, size=n) for i in range(8)}
    data["GDP_Growth"] = 2 + 1.5 * f + rng.normal(scale=0.3, size=n)
    return pd.DataFrame(data, index=idx)


def test_transform():
    s = pd.Series([100.0, 110.0, 121.0])
    assert transform(s, "diff").iloc[1] == 10
    assert np.isclose(transform(s, "dlog").iloc[2], 400 * np.log(1.1))


def test_expanding_window_uses_only_past_data():
    panel = _synthetic_panel()
    models = {"AR(1)": AR1(), "DFM": FactorModel(n_factors=1)}
    out = expanding_window(panel, "GDP_Growth", models, test_start="2015-01-01", verbose=False)
    assert out.index.min() >= pd.Timestamp("2015-01-01")
    # Changing a future value must not change an earlier nowcast.
    panel2 = panel.copy()
    panel2.iloc[-1, :] += 100
    out2 = expanding_window(panel2, "GDP_Growth", models, test_start="2015-01-01", verbose=False)
    pd.testing.assert_frame_equal(out.iloc[:-1], out2.iloc[:-1])


def test_factor_model_beats_ar1_when_factor_drives_gdp():
    panel = _synthetic_panel(n=120)
    models = {"AR(1)": AR1(), "DFM": FactorModel(n_factors=1)}
    out = expanding_window(panel, "GDP_Growth", models, test_start="2020-01-01", verbose=False)
    scores = score_table(out.rename_axis(None))
    full = scores[scores["Period"] == "Full test sample"].set_index("Model")
    assert full.loc["DFM", "RMSE"] < full.loc["AR(1)", "RMSE"]


def test_metrics_perfect_forecast():
    a = pd.Series([1.0, 2.0, 3.0])
    m = metrics(a, a, pd.Series([0.0, 0.0, 0.0]))
    assert m["RMSE"] == 0 and m["MAE"] == 0 and m["MDA"] == 1
