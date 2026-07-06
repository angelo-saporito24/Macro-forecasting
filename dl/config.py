"""Configuration for the MS-GLSTM calibrated forecaster (ST456).

Constants reproduced from the ST456 notebook. This module is import-safe with
no heavy dependencies so the rest of the package (and the tests) can read the
schema without importing TensorFlow.
"""

from __future__ import annotations

from pathlib import Path

# --- Sequence / feature setup --------------------------------------------- #
SEQ_LEN = 24
BATCH_SIZE = 32
MONTHLY_FEATURES = ["pi_mom", "ugap_cf_neg", "VIX", "indpro_ld", "T10Y2Y"]
QUARTERLY_FEATURES = ["gdp_gap_cf"]
ALL_FEATURES = MONTHLY_FEATURES + QUARTERLY_FEATURES
TARGET_COLS = ["pi_mom", "ugap_cf_neg"]  # inflation, (negated) unemployment gap

# Natural (pre-negation) feature set used to fit normalisation parameters.
NORM_FEATURE_COLS = ["pi_mom", "ugap_cf", "VIX", "indpro_ld", "T10Y2Y", "gdp_gap_cf"]

# --- Sample window -------------------------------------------------------- #
START_DATE = "2000-01-01"
END_DATE = "2024-12-31"
TRAIN_END = "2017-12-31"  # normalisation parameters fit on train only

# --- Model architecture --------------------------------------------------- #
UNITS_MONTHLY = 64
UNITS_QUARTERLY = 32
UNITS_FUSED = 48
LOG_VAR_CLAMP = 4.0

# --- Training ------------------------------------------------------------- #
LR = 1e-3
EPOCHS = 100
PATIENCE = 15
GRAD_CLIP_NORM = 1.0
SIGMA_REG_WEIGHT = 0.01
SEED = 42

# lambda grid for the joint NLL+CRPS loss (1.0 = pure NLL, 0.0 = pure CRPS).
LAMBDA_GRID = [1.0, 0.9, 0.75, 0.5, 0.25]

# --- Temporal splits (strictly no lookahead) ------------------------------ #
SPLITS = {
    "train": ("2000-01-01", "2017-12-31"),
    "val": ("2016-01-01", "2019-12-31"),
    "test_pre": ("2013-01-01", "2019-12-31"),
    "test_covid": ("2018-01-01", "2021-12-31"),
    "test_post": ("2020-01-01", "2024-12-31"),
}

# Evaluation regimes (used to report calibration separately per regime).
REGIME_SPANS = {
    "pre_covid":  ("2015-01-01", "2019-12-31"),
    "covid":      ("2020-01-01", "2021-12-31"),
    "post_covid": ("2022-01-01", "2024-12-31"),
}

# --- FRED source series (documented for reproducibility) ------------------ #
# The raw FRED pull and feature engineering live in the ST456 notebook
# (Section 1); this package consumes the cached master_df it produces.
FRED_SERIES = {
    "CPIAUCSL": "monthly",   # -> pi_mom (annualised MoM log-return of CPI)
    "UNRATE": "monthly",     # -> ugap_cf_neg (negated unemployment gap)
    "INDPRO": "monthly",     # -> indpro_ld (log first difference)
    "T10Y2Y": "daily",       # 10Y-2Y spread (monthly mean)
    "VIXCLS": "daily",       # VIX (monthly mean)
    "GDPC1": "quarterly",    # -> gdp_gap_cf (HP-filtered gap, upsampled)
}

# --- Artifact paths ------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent / "data"
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "dl"
WEIGHTS_TEMPLATE = ARTIFACT_DIR / "msglstm_lambda{lam}.weights.h5"
GRID_JSON = ARTIFACT_DIR / "msglstm_grid_results.json"
PREDS_NPZ = ARTIFACT_DIR / "msglstm_predictions.npz"
