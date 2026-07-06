"""Tests for the belief-state loaders (the DL -> RL bridge).

These verify that the Tier-1 loader actually fixes the two flaws the ST455
assessor named: means are genuinely noisy (not near-true), and reported
uncertainty is time-varying. They also check the calibration semantics:
a calibrated forecaster's intervals cover at ~their nominal rate, while an
overconfident forecaster's intervals under-cover.
"""

from __future__ import annotations

import numpy as np

from common.belief import (
    BeliefConfig,
    ForecasterBeliefLoader,
    PlaceholderLoader,
    SIGMA_PI_CALIB,
    make_loader,
)


def _coverage_within_1sigma(loader, pi_path):
    """Fraction of steps where the true pi falls inside mu +/- 1 reported sigma."""
    loader.reset(seed=0)
    inside = 0
    for t, pi in enumerate(pi_path):
        mu_pi, s2_pi, _, _ = loader.get(t, float(pi), 0.0)
        if abs(pi - mu_pi) <= np.sqrt(s2_pi):
            inside += 1
    return inside / len(pi_path)


def _synthetic_pi_path(n=4000, seed=1):
    """Quiet -> turbulent -> quiet inflation path around the 2% target."""
    rng = np.random.default_rng(seed)
    path = []
    for t in range(n):
        turbulent = (n // 3) <= t < (2 * n // 3)
        shock = rng.normal(0, 1.5 if turbulent else 0.3)
        path.append(2.0 + shock)
    return np.array(path)


def test_placeholder_means_are_nearly_true():
    loader = PlaceholderLoader(SIGMA_PI_CALIB, 1.0, noise_scale=0.01, seed=0)
    loader.reset(seed=0)
    errs = [abs(3.0 - loader.get(t, 3.0, 0.0)[0]) for t in range(500)]
    # The flaw: mean error is ~noise_scale, i.e. essentially the true state.
    assert np.mean(errs) < 0.05


def test_tier1_means_are_genuinely_noisy():
    loader = make_loader("calibrated")
    loader.reset(seed=0)
    errs = [abs(3.0 - loader.get(t, 3.0, 0.0)[0]) for t in range(2000)]
    # Mean error should be on the order of the forecaster's own sigma, not ~0.
    assert np.mean(errs) > 0.3 * SIGMA_PI_CALIB


def test_tier1_sigma_is_time_varying():
    loader = make_loader("calibrated")
    loader.reset(seed=0)
    # Quiet stretch at target.
    quiet = [loader.get(t, 2.0, 0.0)[1] for t in range(200)]
    # Turbulent stretch far from target.
    turbulent = [loader.get(t, 6.0, 4.0)[1] for t in range(200)]
    assert np.mean(turbulent) > np.mean(quiet)  # sigma widens under turbulence


def test_placeholder_sigma_is_fixed():
    loader = PlaceholderLoader(SIGMA_PI_CALIB, 1.0, seed=0)
    loader.reset(seed=0)
    s_quiet = loader.get(0, 2.0, 0.0)[1]
    s_turbulent = loader.get(1, 8.0, 8.0)[1]
    assert s_quiet == s_turbulent  # the flaw: sigma does not respond to state


def test_calibrated_covers_near_nominal():
    path = _synthetic_pi_path()
    cov = _coverage_within_1sigma(make_loader("calibrated"), path)
    # ~1 sigma Gaussian coverage is ~0.68; allow a broad band.
    assert 0.60 <= cov <= 0.80


def test_overconfident_undercovers():
    path = _synthetic_pi_path()
    cov_calib = _coverage_within_1sigma(make_loader("calibrated"), path)
    cov_overc = _coverage_within_1sigma(make_loader("overconfident"), path)
    # Overconfident intervals are too tight -> materially lower coverage.
    assert cov_overc < cov_calib
    assert cov_overc < 0.55


def test_reset_is_reproducible():
    loader = make_loader("calibrated")
    loader.reset(seed=123)
    a = [loader.get(t, 2.5, 0.5)[0] for t in range(50)]
    loader.reset(seed=123)
    b = [loader.get(t, 2.5, 0.5)[0] for t in range(50)]
    assert a == b


def test_belief_vector_shape_and_dtype():
    loader = make_loader("calibrated")
    loader.reset(seed=0)
    belief = loader.get(0, 2.0, 0.0)
    assert belief.shape == (4,)
    assert belief.dtype == np.float32
    assert belief[1] > 0 and belief[3] > 0  # variances positive
