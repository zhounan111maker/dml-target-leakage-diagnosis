# Data

## BankChurners.csv

- **Source**: Kaggle — [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) (also mirrored widely as `BankChurners` / credit-card customers)
- **Size**: ~10,000 rows, 20 columns
- **Role in this study**: provides **real covariates only**. Treatment `D` and outcome `y` are simulated (see `src/dgp.py`), which is what makes a known ground-truth ATE possible.

### Placement

Place the file at `data/BankChurners.csv`. All scripts resolve it relative to the repository root.

### Columns used

See `src/dgp.py::CORE_NUMERIC` and `CORE_CATEGORICAL`. The column `Total_Trans_Amt`
plays a special role: it is the **baseline potential outcome** $Y(0)$ on which the
simulated treatment effect acts — and, in the leaky configuration, it is also the
input to one derived feature. That double role is the subject of this study.

### Licensing note

The dataset is distributed via Kaggle under that platform's terms. For research and
teaching use this is standard practice, but if you intend commercial use please
obtain the data directly from Kaggle rather than from this repository.

### Why not use purely synthetic covariates?

Real covariates carry real marginal distributions, correlations and outliers. A
purely synthetic `make_regression`-style setup would understate how easily a
domain-derived feature can accidentally encode the outcome. Keeping $X$ real while
simulating only $D$ and $y$ is the standard compromise in the causal-ML literature.
