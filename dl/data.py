"""Data pipeline for the MS-GLSTM forecaster (TensorFlow-free).

These helpers build the overlapping windowed sequences the dual-stream model
consumes. They depend only on numpy/pandas so they are importable and testable
without TensorFlow.

The raw FRED download and feature engineering (constructing pi_mom, the
Christiano-Fitzgerald / HP-filtered gaps, etc.) live in the ST456 notebook
(Section 1), which writes ``master_df.csv`` and ``norm_params.json``. This
module consumes those cached artifacts; :func:`load_master_df` applies the
train-window normalisation exactly as the notebook does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dl.config import (
    ALL_FEATURES, MONTHLY_FEATURES, QUARTERLY_FEATURES, TARGET_COLS,
    SEQ_LEN, SPLITS,
)

# Column indices of each stream within the ALL_FEATURES tensor.
M_IDX = [ALL_FEATURES.index(c) for c in MONTHLY_FEATURES]
Q_IDX = [ALL_FEATURES.index(c) for c in QUARTERLY_FEATURES]


def load_master_df(data_dir: str | Path) -> pd.DataFrame:
    """Load the cached master DataFrame and apply train-window normalisation.

    Expects ``master_df.csv`` and ``norm_params.json`` (produced by the ST456
    notebook) in ``data_dir``. Returns a normalised copy with ``ugap_cf_neg``.
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "master_df.csv", index_col=0, parse_dates=True)
    with open(data_dir / "norm_params.json") as f:
        norm_params = json.load(f)

    df["ugap_cf_neg"] = -df["ugap_cf"]
    df_norm = df.copy()
    for col in ALL_FEATURES:
        mu, sd = norm_params[col]["mean"], norm_params[col]["std"]
        df_norm[col] = (df[col] - mu) / sd
    return df_norm


def build_sequences(data: pd.DataFrame, feature_cols, target_cols,
                    seq_len: int, start: str, end: str):
    """Build overlapping (X, y, dates) sequences from a date window.

    For each timestep t in [start+seq_len, end], X[t] is the ``seq_len``-month
    history of features ending at t-1, and y[t] is the target at month t.
    """
    # Deduplicate while preserving order: target columns are also features, and
    # duplicate columns would corrupt downstream shapes.
    needed = list(dict.fromkeys(list(feature_cols) + list(target_cols)))
    subset = data.loc[start:end, needed].dropna()
    Xs, ys, ds = [], [], []
    for i in range(seq_len, len(subset)):
        Xs.append(subset[feature_cols].iloc[i - seq_len:i].values)
        ys.append(subset[target_cols].iloc[i].values)
        ds.append(subset.index[i])
    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            pd.DatetimeIndex(ds))


def build_all_splits(df_norm: pd.DataFrame) -> dict:
    """Build sequences for every split in ``SPLITS`` from a normalised df."""
    out = {}
    for name, (s, e) in SPLITS.items():
        X, y, d = build_sequences(df_norm, ALL_FEATURES, TARGET_COLS, SEQ_LEN, s, e)
        out[name] = {"X": X, "y": y, "dates": d}
    return out


def split_streams(X: np.ndarray):
    """Slice the ALL_FEATURES tensor into (monthly, quarterly) sub-tensors."""
    return X[:, :, M_IDX], X[:, :, Q_IDX]
