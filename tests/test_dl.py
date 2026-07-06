"""Tests for the DL module's TensorFlow-free surface.

Covers sequence building, the calibration math, the DL->RL constant provenance,
and that the TF-guarded model raises cleanly when TensorFlow is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl.data import build_sequences, split_streams, M_IDX, Q_IDX
from dl.config import ALL_FEATURES, TARGET_COLS, SEQ_LEN
from dl.calibration import (
    to_natural_units, empirical_coverage, calibration_summary, BRIDGE_CONSTANTS,
)
import common.belief as belief


def _synthetic_df(n=120):
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    cols = list(dict.fromkeys(ALL_FEATURES + TARGET_COLS + ["ugap_cf"]))
    return pd.DataFrame(rng.standard_normal((n, len(cols))), index=idx, columns=cols)


def test_build_sequences_shapes():
    df = _synthetic_df(120)
    X, y, dates = build_sequences(df, ALL_FEATURES, TARGET_COLS, SEQ_LEN,
                                  "2000-01-01", "2009-12-31")
    assert X.ndim == 3 and X.shape[1] == SEQ_LEN and X.shape[2] == len(ALL_FEATURES)
    assert y.shape[0] == X.shape[0] and y.shape[1] == len(TARGET_COLS)
    assert len(dates) == X.shape[0]
    assert X.dtype == np.float32


def test_split_streams_partitions_features():
    df = _synthetic_df(80)
    X, _, _ = build_sequences(df, ALL_FEATURES, TARGET_COLS, SEQ_LEN,
                              "2000-01-01", "2006-12-31")
    X_m, X_q = split_streams(X)
    assert X_m.shape[2] == len(M_IDX)
    assert X_q.shape[2] == len(Q_IDX)
    assert X_m.shape[2] + X_q.shape[2] == len(ALL_FEATURES)


def test_calibration_natural_units_match_bridge():
    # The documented normalised sigma should map to the bridge's natural-unit
    # constants used in common.belief.
    sigma_pi = to_natural_units(BRIDGE_CONSTANTS["calibrated"]["sigma_pi_norm"], "pi_mom")
    sigma_x = to_natural_units(BRIDGE_CONSTANTS["calibrated"]["sigma_x_norm"], "ugap_cf_neg")
    assert sigma_pi == pytest.approx(belief.SIGMA_PI_CALIB, rel=1e-6)
    assert sigma_x == pytest.approx(belief.SIGMA_X_CALIB, rel=1e-6)

    o_pi = to_natural_units(BRIDGE_CONSTANTS["overconfident"]["sigma_pi_norm"], "pi_mom")
    assert o_pi == pytest.approx(belief.SIGMA_PI_OVERC, rel=1e-6)


def test_coverage_and_summary():
    rng = np.random.default_rng(0)
    n = 5000
    y = rng.normal(0, 1, (n, 2))
    mu = np.zeros((n, 2))
    sigma = np.ones((n, 2))  # well-specified -> ~68% within 1 sigma
    cov = empirical_coverage(y[:, 0], mu[:, 0], sigma[:, 0])
    assert 0.63 <= cov <= 0.73
    summ = calibration_summary(y, mu, sigma)
    assert set(summ.keys()) == set(TARGET_COLS)
    assert "sigma_natural" in summ[TARGET_COLS[0]]


def test_msglstm_import_guard():
    # dl.msglstm must import without TensorFlow; building the model without TF
    # should raise a clear ImportError rather than a NameError.
    import dl.msglstm as m
    if not m._HAS_TF:
        with pytest.raises(ImportError):
            m.build_msglstm()
