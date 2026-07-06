"""Tests for the New-Keynesian environment.

Includes the two deterministic validation checks from the ST455 notebook
(stable fixed point, unique attractor under the Taylor rule), Gymnasium API
compliance, and checks that the POMDP observation is wired to the loader.
"""

from __future__ import annotations

import numpy as np
import pytest

from common.belief import make_loader
from rl.env import NKEnv, nk_step, taylor_rule, PI_STAR


# --------------------------------------------------------------------------- #
# Deterministic validation (mirrors notebook Section 3.1)                      #
# --------------------------------------------------------------------------- #
def _roll_taylor(pi0, x0, steps=200):
    pi, x = pi0, x0
    for _ in range(steps):
        i_t = taylor_rule(pi, x)
        pi, x = nk_step(pi, x, i_t)
    return pi, x


def test_stable_fixed_point_under_taylor():
    pi, x = _roll_taylor(PI_STAR, 0.0, steps=200)
    # With beta < 1 the fixed point is displaced from (pi*, 0) by O((1-beta)pi*),
    # but must remain within ~1pp of the labelled steady state.
    assert abs(pi - PI_STAR) < 1.0
    assert abs(x) < 1.0


def test_unique_attractor():
    endpoints = [_roll_taylor(*ic, steps=200)
                 for ic in [(PI_STAR, 0.0), (4.0, 3.0), (0.0, -3.0)]]
    for a in endpoints[1:]:
        assert abs(a[0] - endpoints[0][0]) < 1e-6
        assert abs(a[1] - endpoints[0][1]) < 1e-6


# --------------------------------------------------------------------------- #
# Gymnasium API                                                               #
# --------------------------------------------------------------------------- #
def test_gymnasium_api_compliance():
    from gymnasium.utils.env_checker import check_env
    env = NKEnv(mode="full_obs", ep_len=20)
    check_env(env, skip_render_check=True)


def test_full_obs_shape_and_reset():
    env = NKEnv(mode="full_obs", ep_len=10)
    obs, info = env.reset(seed=0)
    assert obs.shape == (3,)
    assert obs.dtype == np.float32
    np.testing.assert_allclose(obs, [PI_STAR, 0.0, env.R_N + PI_STAR], atol=1e-6)


def test_pomdp_obs_is_five_dim_and_uses_loader():
    env = NKEnv(mode="pomdp_calibrated", loader=make_loader("calibrated"), ep_len=10)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5,)  # [mu_pi, sigma2_pi, mu_x, sigma2_x, i_prev]
    assert obs[1] > 0 and obs[3] > 0  # variance slots are positive


def test_pomdp_requires_loader():
    with pytest.raises(ValueError):
        NKEnv(mode="pomdp_calibrated", loader=None)


def test_action_is_clipped():
    env = NKEnv(mode="full_obs", ep_len=5)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([999.0]))  # way above I_MAX
    assert info["i"] <= env.I_MAX
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([-999.0]))  # below I_MIN
    assert info["i"] >= env.I_MIN


def test_episode_truncates_at_ep_len():
    env = NKEnv(mode="full_obs", ep_len=7)
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            break
    assert steps == 7


def test_taylor_policy_stabilises_stochastic_env():
    """Under the Taylor rule the stochastic economy stays bounded (no divergence)
    and hovers near target."""
    env = NKEnv(mode="full_obs", ep_len=300)
    env.reset(seed=42)
    pis, diverged_any = [], False
    for _ in range(300):
        i_t = env.taylor_action(clip=True)
        _, _, _, trunc, info = env.step(np.array([i_t]))
        pis.append(info["pi"])
        diverged_any |= info["diverged"]
        if trunc:
            break
    assert not diverged_any
    assert abs(np.mean(pis) - PI_STAR) < 1.5
