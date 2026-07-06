"""Calibration: the DL -> RL handoff.

The RL bridge (:mod:`common.belief`) parameterises its belief model with the
forecasters' *measured* uncertainty magnitudes. This module documents where
those numbers come from and recomputes them from saved predictions, so the
constants are reproducible rather than magic.

The values used by the bridge (pre-COVID regime, normalised sigma x target-std,
x100 for inflation to annualised %):

    calibrated   (MS-GLSTM):    sigma_pi ~ 0.6201 * 0.03635 * 100 = 2.254 %
                                sigma_x  ~ 0.5305 * 1.73551       = 0.920 pp
    overconfident (MC Dropout): sigma_pi ~ 0.1255 * 0.03635 * 100 = 0.456 %
                                sigma_x  ~ 0.0906 * 1.73551       = 0.157 pp

The calibrated pair is recovered from the MS-GLSTM predictions npz; the
overconfident pair from the MC Dropout predictions npz (a baseline in the ST456
notebook). ``calibration_from_npz`` computes the mean predicted sigma (and
interval coverage) for any such predictions file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dl.config import TARGET_COLS

# Train-window normalisation std of each target (from norm_params.json).
# Inflation (pi_mom) is additionally scaled x100 to annualised percent.
TARGET_STD = {"pi_mom": 0.03635, "ugap_cf_neg": 1.73551}
TARGET_PCT = {"pi_mom": 100.0, "ugap_cf_neg": 1.0}

# Values consumed by common.belief (mirrored there to keep the RL side
# dependency-free). Kept here as the authoritative provenance.
BRIDGE_CONSTANTS = {
    "calibrated": {"sigma_pi_norm": 0.6201, "sigma_x_norm": 0.5305},
    "overconfident": {"sigma_pi_norm": 0.1255, "sigma_x_norm": 0.0906},
}


def to_natural_units(sigma_norm: float, target: str) -> float:
    """Convert a normalised sigma to natural units (annualised % / pp)."""
    return sigma_norm * TARGET_STD[target] * TARGET_PCT[target]


def empirical_coverage(y, mu, sigma, z: float = 1.0) -> float:
    """Fraction of observations inside mu +/- z*sigma (z=1 -> nominal ~68%)."""
    y, mu, sigma = np.asarray(y), np.asarray(mu), np.asarray(sigma)
    return float(np.mean(np.abs(y - mu) <= z * sigma))


def calibration_summary(y, mu, sigma) -> dict:
    """Per-target mean sigma (normalised + natural) and ~1-sigma coverage."""
    y, mu, sigma = np.asarray(y), np.asarray(mu), np.asarray(sigma)
    out = {}
    for i, target in enumerate(TARGET_COLS):
        s_norm = float(np.mean(sigma[:, i]))
        out[target] = {
            "sigma_norm": s_norm,
            "sigma_natural": to_natural_units(s_norm, target),
            "coverage_1sigma": empirical_coverage(y[:, i], mu[:, i], sigma[:, i]),
        }
    return out


def calibration_from_npz(path: str | Path, regime: str = "test_pre") -> dict:
    """Recompute the calibration summary from a saved predictions npz.

    Expects arrays ``{regime}_mu``, ``{regime}_sigma``, ``{regime}_y``
    (shape (N, n_targets)) as written by the ST456 evaluation.
    """
    data = np.load(Path(path), allow_pickle=True)
    mu = data[f"{regime}_mu"]
    sigma = data[f"{regime}_sigma"]
    y = data[f"{regime}_y"]
    return calibration_summary(y, mu, sigma)
