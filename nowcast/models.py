"""Nowcasting models sharing a fit(X, y) / predict(X) interface.

Every model is refit inside each step of the expanding-window evaluation, so
scaling, imputation and hyperparameter tuning only ever see past data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class AR1(BaseEstimator, RegressorMixin):
    """AR(1) benchmark: y_t = c + phi * y_{t-1}. Ignores the indicators.

    It needs the target history, so `fit` and `predict` take it through the
    `y_lag` column that `evaluate.expanding_window` adds to X.
    """

    def fit(self, X, y):
        self.lr_ = LinearRegression().fit(X[["y_lag"]], y)
        return self

    def predict(self, X):
        return self.lr_.predict(X[["y_lag"]])


class FactorModel(BaseEstimator, RegressorMixin):
    """Two-step dynamic factor model (Stock & Watson diffusion index).

    Step 1 extracts k principal-component factors from the standardised
    indicators; step 2 regresses GDP growth on the factors and its own lag.
    """

    def __init__(self, n_factors: int = 3):
        self.n_factors = n_factors

    def fit(self, X, y):
        ind = X.drop(columns="y_lag")
        self.prep_ = make_pipeline(SimpleImputer(), StandardScaler(),
                                   PCA(n_components=self.n_factors)).fit(ind)
        self.reg_ = LinearRegression().fit(self._design(X), y)
        return self

    def _design(self, X):
        f = self.prep_.transform(X.drop(columns="y_lag"))
        return np.column_stack([f, X["y_lag"].to_numpy()])

    def predict(self, X):
        return self.reg_.predict(self._design(X))


def _scaled(model):
    return make_pipeline(SimpleImputer(), StandardScaler(), model)


def build_models(seed: int = 42) -> dict[str, BaseEstimator]:
    cv = TimeSeriesSplit(n_splits=5)
    return {
        "AR(1)": AR1(),
        "DFM (3 factors)": FactorModel(n_factors=3),
        "OLS": _scaled(LinearRegression()),
        "Ridge": _scaled(RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "Lasso": _scaled(LassoCV(cv=cv, max_iter=50_000, random_state=seed)),
        "Elastic Net": _scaled(ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=cv,
                                            max_iter=50_000, random_state=seed)),
        "SVR": _scaled(SVR(C=10, epsilon=0.5)),
        "Random Forest": _scaled(RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                       random_state=seed, n_jobs=-1)),
        "Gradient Boosting": _scaled(GradientBoostingRegressor(n_estimators=300, max_depth=2,
                                                               learning_rate=0.05,
                                                               subsample=0.8,
                                                               random_state=seed)),
        "MLP": _scaled(MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0, max_iter=5_000,
                                    early_stopping=False, random_state=seed)),
    }


def fit_kalman_dfm(panel: pd.DataFrame, target: str, k_factors: int = 2) -> pd.Series:
    """In-sample Kalman-filter DFM (statsmodels) with GDP growth as an observable.

    Returns fitted GDP growth on the original scale. In-sample only: shown to
    illustrate the state-space approach, not used for the forecast comparison.
    """
    from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

    mu, sd = panel.mean(), panel.std()
    z = (panel - mu) / sd
    res = DynamicFactor(z, k_factors=k_factors, factor_order=1).fit(disp=False, maxiter=2_000)
    fitted = res.predict()[target]
    return fitted * sd[target] + mu[target]


def clone_model(m):
    return clone(m)
