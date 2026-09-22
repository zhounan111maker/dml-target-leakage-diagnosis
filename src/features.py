"""
Derived feature construction with a controllable leakage channel.

Six features are produced: five semantic descriptors plus an aggregate
dispersion index. Five of them are always safe. The remaining one,
`f_wood`, is parameterised by a leakage intensity alpha:

    alpha = 0  ->  f_wood computed WITHOUT the outcome      (clean)
    alpha = 1  ->  f_wood computed FROM the baseline outcome (leaky)

At alpha = 1 the feature is a deterministic function of Y(0), the untreated
potential outcome, which is unobservable by construction. Accessing it is
equivalent to reading the answer key -- see paper.md, Section 3.

The intermediate values of alpha are used for the dose-response sweep: they
blend the two variants *after* standardisation and re-standardise, so that the
marginal distribution of the feature stays comparable across the range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# The six columns this module emits, in canonical order.
FEATURES = ['f_wood', 'f_fire', 'f_earth', 'f_metal', 'f_water', 'f_balance']

INCOME_MAP = {
    'Income_Category_Less than $40K': 1,
    'Income_Category_$40K - $60K': 2,
    'Income_Category_$60K - $80K': 3,
    'Income_Category_$80K - $120K': 4,
    'Income_Category_$120K +': 5,
    'Income_Category_Unknown': 3,
}
CARD_MAP = {
    'Card_Category_Blue': 1,
    'Card_Category_Silver': 2,
    'Card_Category_Gold': 3,
    'Card_Category_Platinum': 4,
}


def zscore(x) -> np.ndarray:
    return StandardScaler().fit_transform(np.asarray(x).reshape(-1, 1)).ravel()


def build_features(X: pd.DataFrame, y0_values, alpha: float) -> pd.DataFrame:
    """
    Build the derived feature block.

    Parameters
    ----------
    X : pd.DataFrame
        Raw covariates (must NOT contain the outcome column).
    y0_values : array-like
        Baseline potential outcome Y(0) for the rows of X. Only consulted when
        alpha > 0 -- this is the leakage channel.
    alpha : float in [0, 1]
        Leakage intensity.

    Returns
    -------
    pd.DataFrame with exactly the columns in FEATURES.
    """
    d = X.copy()

    # -- f_wood : the leakage channel ---------------------------------------
    # clean variant: transaction intensity, no outcome involved
    w_clean = d['Total_Trans_Ct'] / (d['Months_on_book'] + 1e-6)
    # leaky variant: weighted transaction value, built from Y(0)
    w_leaky = ((np.asarray(y0_values) / (d['Total_Trans_Ct'] + 1))
               * np.log1p(d['Total_Trans_Ct']))
    d['f_wood'] = zscore((1 - alpha) * zscore(w_clean) + alpha * zscore(w_leaky))

    # -- f_fire : activity ratio --------------------------------------------
    d['f_fire'] = zscore(np.clip(
        (d['Months_on_book'] - d['Months_Inactive_12_mon']) / d['Months_on_book'],
        0, 1))

    # -- f_earth : credit resource utilisation -------------------------------
    d['f_earth'] = zscore(d['Total_Revolving_Bal'] / (d['Credit_Limit'] + 1e-6))

    # -- f_metal : income / card-tier match ----------------------------------
    inc = sum(d[c] * v for c, v in INCOME_MAP.items() if c in d.columns)
    card = sum(d[c] * v for c, v in CARD_MAP.items() if c in d.columns)
    d['f_metal'] = zscore(1 - (abs(inc - card) / 4))

    # -- f_water : spending-flow stability -----------------------------------
    d['f_water'] = zscore(np.clip(1 - abs(d['Total_Amt_Chng_Q4_Q1'] - 1), 0, 1))

    # -- f_balance : inverse coefficient of variation across the five --------
    core = ['f_wood', 'f_fire', 'f_earth', 'f_metal', 'f_water']
    mean_, std_ = d[core].mean(axis=1), d[core].std(axis=1)
    d['f_balance'] = zscore(1 - (std_ / (mean_ + 1e-6))) * 1.5

    return d[FEATURES]
