"""Training: PPO / SAC central-bank agents on the NK POMDP.

Refactored from the ST455 notebook, with two deliberate, documented changes:

1. The action space is normalised to [-1, 1] via ``RescaleAction`` (SB3-
   recommended, and what the report's own alternative specification used),
   rather than the raw [1, 20]% space. The policy's initial action mean is
   biased toward the steady-state rate i* for stable early training.

2. The belief source is selectable via ``belief_mode``:
     * "placeholder" - the ORIGINAL stub (near-true means, fixed sigma).
     * "tier1"       - the forecaster-informed belief model (noisy means,
                       time-varying, calibration-aware sigma).
   This lets the repo reproduce the original result and the extension for a
   like-for-like comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.env import NKEnv
from common.belief import make_loader

# Conditions (the analytic Taylor rule is a non-trained benchmark).
CONDITIONS = ["taylor_rule", "full_obs_rl", "pomdp_calibrated", "pomdp_overconfident"]
TRAINABLE = ["full_obs_rl", "pomdp_calibrated", "pomdp_overconfident"]

PPO_KWARGS = dict(
    learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
    vf_coef=0.5, max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
    verbose=0, device="cpu",
)

SAC_KWARGS = dict(
    learning_rate=3e-4, buffer_size=100_000, learning_starts=1_000,
    batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
    ent_coef="auto", policy_kwargs=dict(net_arch=[64, 64]),
    verbose=0, device="cpu",
)

# Full-budget defaults (as in the report). Override for quick runs.
PPO_TOTAL_STEPS = 1_000_000
SAC_TOTAL_STEPS = 500_000
SEEDS = [42, 123, 7]
EP_LEN = 20


# --------------------------------------------------------------------------- #
# Environment construction                                                     #
# --------------------------------------------------------------------------- #
def _loader_for(condition: str, belief_mode: str, seed: int):
    """Return the belief loader for a POMDP condition, or None."""
    if condition == "pomdp_calibrated":
        kind = "calibrated" if belief_mode == "tier1" else "placeholder_calib"
    elif condition == "pomdp_overconfident":
        kind = "overconfident" if belief_mode == "tier1" else "placeholder_overconf"
    else:
        return None
    if belief_mode == "tier1":
        return make_loader(kind, seed=seed)
    return make_loader(kind, seed=seed)  # placeholder_* also accept seed


def make_env(condition: str, belief_mode: str = "tier1",
             ep_len: int = EP_LEN, seed: int = 42) -> gym.Env:
    """Build a single (RescaleAction + Monitor)-wrapped NK environment."""
    mode = {"full_obs_rl": "full_obs",
            "pomdp_calibrated": "pomdp_calibrated",
            "pomdp_overconfident": "pomdp_overconf"}[condition]
    loader = _loader_for(condition, belief_mode, seed)
    env = NKEnv(mode=mode, ep_len=ep_len, seed=seed, loader=loader)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    return Monitor(env)


def _bias_policy_toward_steady_state(model, i_star=NKEnv.R_N + NKEnv.PI_STAR):
    """Initialise the policy's action mean near i* in the normalised [-1,1] space.

    With RescaleAction the action is an affine map of the policy output, so the
    target normalised action is a simple linear rescaling of i* (no tanh).
    """
    low, high = NKEnv.I_MIN, NKEnv.I_MAX
    a_norm = float(np.clip(2.0 * (i_star - low) / (high - low) - 1.0, -0.999, 0.999))
    with torch.no_grad():
        model.policy.action_net.bias.fill_(a_norm)


class EpisodeLogCallback(BaseCallback):
    """Log episodic return every ``log_freq`` completed episodes."""

    def __init__(self, log_freq=10):
        super().__init__()
        self.log_freq = log_freq
        self.records = []
        self._ep_count = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_count += 1
                if self._ep_count % self.log_freq == 0:
                    self.records.append({
                        "episode": self._ep_count,
                        "mean_reward": info["episode"]["r"],
                        "timestep": self.num_timesteps,
                    })
        return True

    def as_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.records)


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_agent(algo: str, condition: str, belief_mode: str, total_steps: int,
                seed: int, out_dir: str | Path, overwrite: bool = False):
    """Train one agent for one (algo, condition, seed); save model + log.

    Returns the path to the saved ``model.zip`` (cached if it already exists).
    """
    algo = algo.lower()
    out_dir = Path(out_dir) / f"{algo}_{belief_mode}_{condition}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.zip"
    if model_path.exists() and not overwrite:
        return model_path

    env = make_env(condition, belief_mode, seed=seed)
    vec_env = DummyVecEnv([lambda: env])
    cb = EpisodeLogCallback(log_freq=10)

    if algo == "ppo":
        model = PPO("MlpPolicy", vec_env, **{**PPO_KWARGS, "seed": seed})
        _bias_policy_toward_steady_state(model)
    elif algo == "sac":
        model = SAC("MlpPolicy", vec_env, **{**SAC_KWARGS, "seed": seed})
    else:
        raise ValueError(f"Unknown algo {algo!r} (use 'ppo' or 'sac').")

    model.learn(total_timesteps=total_steps, callback=cb)
    model.save(str(out_dir / "model"))
    log = cb.as_dataframe()
    if not log.empty:
        log.to_csv(out_dir / "training_log.csv", index=False)
    return model_path
