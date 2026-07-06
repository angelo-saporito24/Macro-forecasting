"""Evaluation: deterministic rollouts, central-bank loss, Taylor projection.

Mirrors the ST455 evaluation protocol:

* Each agent is rolled out deterministically (policy mean) over ``n_episodes``
  episodes of length ``ep_len``.
* The belief-state loader is re-initialised with ``seed + 1`` so evaluation
  beliefs are drawn from a different sequence than training.
* The reported reward is the *raw* per-episode central-bank loss
  -sum_t[(pi - pi*)^2 + lam * x^2], not the normalised training reward.
* The Taylor-principle test regresses the agent's rate on its (belief) state,
  reporting the long-run inflation response phi_pi_lr = phi_pi / (1 - rho).
"""

from __future__ import annotations

import numpy as np
from gymnasium.wrappers import RescaleAction
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from rl.env import NKEnv, PI_STAR
from rl.train import EP_LEN, _loader_for

I_STEADY = NKEnv.R_N + NKEnv.PI_STAR


def _eval_env(condition: str, belief_mode: str, seed: int, ep_len: int):
    """Eval environment (RescaleAction-wrapped) with the loader seeded seed+1."""
    mode = {"full_obs_rl": "full_obs",
            "pomdp_calibrated": "pomdp_calibrated",
            "pomdp_overconfident": "pomdp_overconf"}[condition]
    loader = _loader_for(condition, belief_mode, seed + 1)
    env = NKEnv(mode=mode, ep_len=ep_len, seed=seed, loader=loader)
    return RescaleAction(env, min_action=-1.0, max_action=1.0)


def rollout_eval(model, condition: str, belief_mode: str = "tier1",
                 n_episodes: int = 20, seed: int = 42, ep_len: int = EP_LEN):
    """Roll out a policy and return per-step arrays.

    ``model`` is ignored for the ``taylor_rule`` condition (analytic policy).
    Returns a dict with pi, x, i, i_lag, reward, mu_pi, mu_x.
    """
    pi_all, x_all, i_all, r_all, mu_pi_all, mu_x_all = [], [], [], [], [], []
    diverged = 0

    if condition == "taylor_rule":
        env = NKEnv(mode="full_obs", ep_len=ep_len, seed=seed)
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                i_t = env.taylor_action(clip=False)
                obs, _, term, trunc, info = env.step(np.array([i_t]))
                pi_all.append(info["pi"]); x_all.append(info["x"])
                i_all.append(i_t); r_all.append(info["reward"])
                mu_pi_all.append(info["pi"]); mu_x_all.append(info["x"])
                done = term or trunc
    else:
        env = _eval_env(condition, belief_mode, seed, ep_len)
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                pi_all.append(info["pi"]); x_all.append(info["x"])
                i_all.append(info["i"]); r_all.append(info["reward"])
                diverged += int(info["diverged"])
                # Read the belief the agent actually observed from the obs.
                if len(obs) == 3:  # full_obs: [pi, x, i_prev]
                    mu_pi_all.append(float(obs[0])); mu_x_all.append(float(obs[1]))
                else:              # pomdp: [mu_pi, s2, mu_x, s2, i_prev]
                    mu_pi_all.append(float(obs[0])); mu_x_all.append(float(obs[2]))
                done = term or trunc

    i_arr = np.array(i_all)
    n = len(i_arr)
    i_lag = np.empty_like(i_arr)
    for start in range(0, n, ep_len):
        i_lag[start] = I_STEADY
        end = min(start + ep_len, n)
        i_lag[start + 1:end] = i_arr[start:end - 1]

    return dict(pi=np.array(pi_all), x=np.array(x_all), i=i_arr, i_lag=i_lag,
                reward=np.array(r_all), mu_pi=np.array(mu_pi_all),
                mu_x=np.array(mu_x_all), diverged=diverged,
                n_episodes=n_episodes)


def taylor_projection(res: dict, condition: str) -> dict:
    """OLS projection of the policy onto the Taylor-rule basis (HC3 SEs)."""
    use_true = condition in ("full_obs_rl", "taylor_rule")
    pi_input = (res["pi"] if use_true else res["mu_pi"]) - PI_STAR
    x_input = res["x"] if use_true else res["mu_x"]

    X = add_constant(np.column_stack([res["i_lag"], pi_input, x_input]))
    ols = OLS(res["i"], X).fit(cov_type="HC3")
    rho, phi_pi, phi_y = ols.params[1], ols.params[2], ols.params[3]
    denom = (1 - rho) if abs(1 - rho) > 0.01 else np.nan
    return dict(rho=float(rho), phi_pi=float(phi_pi),
                phi_pi_lr=float(phi_pi / denom), phi_y_lr=float(phi_y / denom),
                p_phi_pi=float(ols.pvalues[2]), r2=float(ols.rsquared))


def summarize(res: dict) -> dict:
    """Headline metrics for one condition."""
    mean_ep_reward = float(res["reward"].sum() / res["n_episodes"])
    return dict(
        mean_episode_reward=mean_ep_reward,
        mean_pi=float(res["pi"].mean()),
        mean_x=float(res["x"].mean()),
        mean_i=float(res["i"].mean()),
        diverged_steps=int(res["diverged"]),
    )


def evaluate_condition(model, condition: str, belief_mode: str = "tier1",
                       n_episodes: int = 20, seed: int = 42,
                       ep_len: int = EP_LEN) -> dict:
    """Roll out, summarise, and project one condition in one call."""
    res = rollout_eval(model, condition, belief_mode, n_episodes, seed, ep_len)
    out = summarize(res)
    if len(res["i"]) > 5:
        out.update(taylor_projection(res, condition))
    return out


def results_table(rows: dict):
    """Assemble a comparison DataFrame from {condition_label: metrics} rows."""
    import pandas as pd
    df = pd.DataFrame(rows).T
    preferred = ["mean_episode_reward", "phi_pi_lr", "rho", "r2",
                 "mean_pi", "mean_x", "mean_i", "diverged_steps"]
    cols = [c for c in preferred if c in df.columns]
    return df[cols]
