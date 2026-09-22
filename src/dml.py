"""
Double Machine Learning estimator (Chernozhukov et al., 2018).

2-fold cross-fitting with LightGBM as both nuisance learners:

    Y - g(X) = theta * (D - m(X)) + eps

  step 1  fit g_hat(X) ~ E[Y|X] and m_hat(X) ~ E[D|X] on the training fold
  step 2  compute out-of-fold residuals  Y - g_hat(X)  and  D - m_hat(X)
  step 3  OLS of the residualised outcome on the residualised treatment

Deliberately minimal. The paper's contribution concerns feature construction,
so the estimator is held fixed and simple across all arms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.model_selection import KFold

LGB_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def dml_estimate(X, D, y, n_splits: int = 2, seed: int = 42):
    """
    Returns
    -------
    (ate, se) : tuple[float, float]
        Point estimate of the ATE and its OLS standard error.
    """
    X = pd.DataFrame(X).reset_index(drop=True)
    D = pd.Series(D).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_res = np.zeros(len(y), dtype=float)
    d_res = np.zeros(len(D), dtype=float)

    model_y = lgb.LGBMRegressor(**LGB_PARAMS)
    model_d = lgb.LGBMRegressor(**LGB_PARAMS)

    for tr, te in kf.split(X):
        model_y.fit(X.iloc[tr], y.iloc[tr])
        model_d.fit(X.iloc[tr], D.iloc[tr])
        y_res[te] = y.iloc[te].to_numpy() - model_y.predict(X.iloc[te])
        d_res[te] = D.iloc[te].to_numpy() - model_d.predict(X.iloc[te])

    ols = sm.OLS(y_res, sm.add_constant(d_res)).fit()
    return float(np.asarray(ols.params).ravel()[-1]), \
           float(np.asarray(ols.bse).ravel()[-1])


def dml_ensemble(X, D, y, n_splits: int = 2, seed: int = 42):
    """
    Variant averaging LightGBM and XGBoost predictions for both nuisances.

    Reported in the paper as producing substantively identical conclusions;
    provided for completeness.
    """
    import xgboost as xgb

    X = pd.DataFrame(X).reset_index(drop=True)
    D = pd.Series(D).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    xgb_params = dict(n_estimators=100, learning_rate=0.05, max_depth=5,
                      reg_lambda=1.5, reg_alpha=0.5, random_state=42,
                      n_jobs=-1, verbosity=0)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_res = np.zeros(len(y), dtype=float)
    d_res = np.zeros(len(D), dtype=float)

    for tr, te in kf.split(X):
        for target, out in ((y, y_res), (D, d_res)):
            m1 = lgb.LGBMRegressor(**LGB_PARAMS)
            m2 = xgb.XGBRegressor(**xgb_params)
            m1.fit(X.iloc[tr], target.iloc[tr])
            m2.fit(X.iloc[tr], target.iloc[tr])
            out[te] = (target.iloc[te].to_numpy()
                       - (m1.predict(X.iloc[te]) + m2.predict(X.iloc[te])) / 2)

    ols = sm.OLS(y_res, sm.add_constant(d_res)).fit()
    return float(np.asarray(ols.params).ravel()[-1]), \
           float(np.asarray(ols.bse).ravel()[-1])
