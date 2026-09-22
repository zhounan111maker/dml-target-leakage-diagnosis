"""
Semi-synthetic data generating process.

Real covariates from the Kaggle Bank Customer Churn dataset; treatment and
outcome simulated with a known, homogeneous, multiplicative causal effect.

    p(D=1 | X) = clip(0.7 * CreditLimit/max + 0.3 * MonthsInactive/max, 0.2, 0.8)
    Y          = Y(0) * (1 + tau_mult - 1)^D
    tau        = E[Y(1) - Y(0)] = (m - 1) * E[Y(0)]        over the FULL sample

Note that D is a function of CreditLimit and MonthsInactive, which are
themselves correlated with the baseline outcome. The two arms therefore have
different baseline outcome means, and E[Y(0) | D=0] is NOT an unbiased
substitute for E[Y(0)]. See `true_ate_control_only` and `naive_difference`
for the two incorrect estimands this paper documents.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TREAT_MULT = 1.15

CORE_NUMERIC = [
    'Customer_Age', 'Dependent_count', 'Months_on_book', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Avg_Utilization_Ratio',
]
CORE_CATEGORICAL = ['Card_Category', 'Income_Category']

OUTCOME_BASE = 'Total_Trans_Amt'


def load_data(path: str = 'data/BankChurners.csv') -> pd.DataFrame:
    """Load and one-hot encode the covariates. Returns a DataFrame of length n."""
    df = pd.read_csv(path)
    core = df[CORE_NUMERIC + CORE_CATEGORICAL].dropna()
    return pd.get_dummies(
        core, columns=CORE_CATEGORICAL, drop_first=False
    ).reset_index(drop=True)


def simulate_treatment(df_encoded: pd.DataFrame, rng: np.random.RandomState) -> pd.Series:
    """Treatment assignment. Confounded by construction."""
    credit_norm = df_encoded['Credit_Limit'] / df_encoded['Credit_Limit'].max()
    inactive_norm = (df_encoded['Months_Inactive_12_mon']
                     / df_encoded['Months_Inactive_12_mon'].max())
    offer_prob = np.clip(0.7 * credit_norm + 0.3 * inactive_norm, 0.2, 0.8)
    D = (offer_prob > rng.uniform(0, 1, len(df_encoded))).astype(int)
    return pd.Series(D, name='D')


def simulate_outcome(df_encoded: pd.DataFrame, D: pd.Series,
                     mult: float = TREAT_MULT) -> pd.Series:
    """Y = Y(0) * mult^D, so Y(1) - Y(0) = (mult - 1) * Y(0) for every unit."""
    y0 = df_encoded[OUTCOME_BASE]
    return pd.Series(np.where(D == 1, y0 * mult, y0), name='y')


def true_ate_full(y0: pd.Series, mult: float = TREAT_MULT) -> float:
    """The correct estimand: (mult - 1) * E[Y(0)] over the full sample."""
    return float((mult - 1.0) * y0.mean())


def true_ate_control_only(y0: pd.Series, D: pd.Series,
                          mult: float = TREAT_MULT) -> float:
    """INCORRECT. Uses only the control-group mean of Y(0).

    Reproduces the original pipeline's error. Introduces a small negative bias
    because the control arm has a lower baseline outcome than the full sample.
    """
    return float((mult - 1.0) * y0[D == 0].mean())


def naive_difference(y: pd.Series, D: pd.Series) -> float:
    """INCORRECT as an estimand. Naive difference in means, confounded.

    This is the quantity a causal estimator exists to correct; adopting it as
    'theoretical truth' rewards estimators that remove LESS confounding.
    """
    return float(y[D == 1].mean() - y[D == 0].mean())


def ground_truth_table(df_encoded: pd.DataFrame, D: pd.Series,
                       y: pd.Series, mult: float = TREAT_MULT) -> dict:
    """Every quantity in Table 1 of the paper."""
    y0 = df_encoded[OUTCOME_BASE]
    correct = true_ate_full(y0, mult)
    wrong = true_ate_control_only(y0, D, mult)
    naive = naive_difference(y, D)
    return {
        'E[Y0]_full': round(float(y0.mean()), 2),
        'E[Y0]_D0': round(float(y0[D == 0].mean()), 2),
        'E[Y0]_D1': round(float(y0[D == 1].mean()), 2),
        'ate_correct': round(correct, 2),
        'ate_control_mean_error': round(wrong, 2),
        'naive_difference': round(naive, 2),
        'rel_err_control_mean_pct': round((wrong / correct - 1) * 100, 2),
        'rel_err_naive_pct': round((naive / correct - 1) * 100, 2),
        'arm_imbalance_pct': round(
            (y0[D == 1].mean() / y0[D == 0].mean() - 1) * 100, 2),
    }
