"""Fast end-to-end smoke test for the RL train + eval pipeline.

Trains a tiny PPO agent and evaluates it, so CI exercises the whole path
(env -> loader -> SB3 -> eval -> regression) without a long training run.
"""

from __future__ import annotations

import math

from rl.train import train_agent
from rl.evaluate import evaluate_condition
from rl.run import _load_model


def test_ppo_pipeline_smoke(tmp_path):
    out = tmp_path / "rl"
    # One short rollout's worth of training on the full-obs condition.
    path = train_agent("ppo", "full_obs_rl", "tier1", total_steps=1000,
                       seed=0, out_dir=out)
    assert path.exists()

    model = _load_model("ppo", "full_obs_rl", "tier1", 0, out)
    metrics = evaluate_condition(model, "full_obs_rl", "tier1",
                                 n_episodes=3, seed=0)
    for key in ("mean_episode_reward", "mean_pi", "phi_pi_lr", "rho", "r2"):
        assert key in metrics
        assert math.isfinite(metrics[key])


def test_pomdp_eval_reads_belief(tmp_path):
    out = tmp_path / "rl"
    train_agent("ppo", "pomdp_calibrated", "tier1", total_steps=1000,
                seed=0, out_dir=out)
    model = _load_model("ppo", "pomdp_calibrated", "tier1", 0, out)
    m = evaluate_condition(model, "pomdp_calibrated", "tier1",
                           n_episodes=3, seed=0)
    assert math.isfinite(m["mean_episode_reward"])


def test_taylor_baseline_needs_no_model():
    m = evaluate_condition(None, "taylor_rule", "tier1", n_episodes=10, seed=0)
    # The analytic Taylor rule respects the Taylor principle. In the inertial
    # regression the response loads onto the LONG-RUN coefficient (the short-run
    # one is split with a spurious rho), so we check phi_pi_lr > 1.
    assert m["phi_pi_lr"] > 1.0
