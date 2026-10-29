"""Run the US GDP nowcasting comparison end to end.

    python run_nowcast.py                 # uses cached data/fred_raw.csv if present
    python run_nowcast.py --refresh       # re-download every series from FRED
    python run_nowcast.py --kalman-dfm    # also fit the in-sample state-space DFM
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

from nowcast.data import TARGET, build_quarterly_panel
from nowcast.evaluate import expanding_window, score_table
from nowcast.models import build_models, fit_kalman_dfm

warnings.simplefilter("ignore", ConvergenceWarning)


def plot_nowcasts(nowcasts, models, out: Path) -> None:
    for name in models:
        if name == "AR(1)":
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(nowcasts.index, nowcasts["Actual"], color="black", marker="o", ms=3,
                label="Actual GDP growth")
        ax.plot(nowcasts.index, nowcasts[name], linestyle="--", marker="o", ms=3,
                label=f"{name} nowcast")
        ax.plot(nowcasts.index, nowcasts["AR(1)"], color="grey", alpha=0.6, label="AR(1) benchmark")
        ax.set_title(f"{name}: out-of-sample nowcast vs actual US GDP growth")
        ax.set_ylabel("Real GDP growth (% SAAR)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(out / f"nowcast_{slug}.png", dpi=150)
        plt.close(fig)


def plot_rmse(scores, out: Path) -> None:
    pivot = scores.pivot(index="Model", columns="Period", values="RMSE vs AR(1)")
    pivot = pivot.sort_values("Full test sample")
    ax = pivot.plot.barh(figsize=(10, 6))
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("RMSE relative to AR(1)  (< 1 beats the benchmark)")
    ax.set_title("Out-of-sample nowcast accuracy")
    plt.tight_layout()
    plt.savefig(out / "relative_rmse.png", dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--start", default="2001-01-01",
                   help="first quarter of the estimation sample (Job Openings starts Dec 2000)")
    p.add_argument("--test-start", default="2012-01-01")
    p.add_argument("--refresh", action="store_true", help="re-download data from FRED")
    p.add_argument("--kalman-dfm", action="store_true")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Building quarterly panel...")
    panel = build_quarterly_panel(start=args.start, refresh=args.refresh)
    print(f"Panel: {panel.shape[0]} quarters x {panel.shape[1]} series, "
          f"{panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}")

    models = build_models()
    print(f"Expanding-window nowcasts from {args.test_start}...")
    nowcasts = expanding_window(panel, TARGET, models, test_start=args.test_start)
    nowcasts.to_csv(args.out / "nowcasts.csv")

    scores = score_table(nowcasts)
    scores.to_csv(args.out / "scores.csv", index=False)
    for period, tbl in scores.groupby("Period", sort=False):
        print(f"\n{period}  (N = {tbl['N'].iloc[0]})")
        print(tbl.drop(columns=["Period", "N"]).sort_values("RMSE").to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))

    plot_nowcasts(nowcasts, models, args.out)
    plot_rmse(scores, args.out)

    if args.kalman_dfm:
        print("\nFitting in-sample Kalman-filter DFM...")
        fitted = fit_kalman_dfm(panel.dropna(), TARGET)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(panel.index, panel[TARGET], color="black", label="Actual")
        ax.plot(fitted.index, fitted, linestyle="--", label="State-space DFM (in-sample)")
        ax.set_ylabel("Real GDP growth (% SAAR)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out / "kalman_dfm_in_sample.png", dpi=150)
        plt.close(fig)

    print(f"\nResults written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
