"""Three-equation New-Keynesian economy as a Gymnasium environment.

Refactored from the ST455 notebook. Units are annualised percentage points
(a 2% target is ``2.0``). The agent is a central bank that sets the nominal
policy rate each period to stabilise inflation and the output gap.

Observability is controlled by ``mode``:

* ``full_obs``         - the agent observes the true state [pi, x, i_prev].
* ``pomdp_calibrated`` - the agent observes a *belief* about the state, supplied
* ``pomdp_overconf``     by a :class:`~common.belief.BeliefStateLoader`. The two
                         POMDP modes are identical in dynamics; which one is
                         "calibrated" vs "overconfident" is a property of the
                         loader that is passed in, not of the environment.

The belief observation is [mu_pi, sigma2_pi, mu_x, sigma2_x, i_prev].
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from common.belief import BeliefStateLoader


class NKEnv(gym.Env):
    """New-Keynesian model with hybrid expectations and a divergence penalty.

    Transition (hybrid expectations, omega=0.7):
        IS:   x_{t+1}  = x_t - (1/sigma)(i_t - E[pi_{t+1}] - r^n) + eps_d
        NKPC: pi_{t+1} = beta * E[pi_{t+1}] + kappa * x_t + eps_s

    Reward (normalised quadratic central-bank loss):
        r_t = -[(pi_dev/pi_scale)^2 + lam * (x/x_scale)^2]
              minus a large penalty if the state approaches the clip boundary.
    """

    # Structural parameters
    BETA = 0.99;   SIGMA = 1.0;    KAPPA = 0.05
    PI_STAR = 2.0; R_N = 4.0;      LAM = 0.5
    RHO_D = 0.5;   RHO_S = 0.5
    SIGMA_D = 0.4; SIGMA_S = 0.2

    # Taylor-rule coefficients (used by the analytic benchmark policy)
    PHI_PI = 1.5;  PHI_Y = 0.5

    # Bounds / scaling
    I_MIN = 1.0;   I_MAX = 20.0
    PI_CLIP = 6.0; X_CLIP = 10.0
    PI_SCALE = 1.0; X_SCALE = 2.0
    DIV_PENALTY = 200.0

    OMEGA = 0.7  # weight on current inflation in hybrid expectations

    def __init__(self, mode: str = "full_obs",
                 loader: Optional[BeliefStateLoader] = None,
                 ep_len: int = 50, seed: int = 42, verbose: bool = False):
        super().__init__()
        assert mode in ("full_obs", "pomdp_calibrated", "pomdp_overconf")
        if mode != "full_obs" and loader is None:
            raise ValueError(f"mode='{mode}' requires a BeliefStateLoader.")

        self.mode = mode
        self.loader = loader
        self.ep_len = ep_len
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(
            low=self.I_MIN, high=self.I_MAX, shape=(1,), dtype=np.float32)

        if mode == "full_obs":
            obs_low = np.array([-10.0, -20.0, 0.0], dtype=np.float32)
            obs_high = np.array([20.0, 20.0, 20.0], dtype=np.float32)
        else:
            obs_low = np.array([-10.0, 0.0, -20.0, 0.0, 0.0], dtype=np.float32)
            obs_high = np.array([20.0, 50.0, 20.0, 50.0, 20.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.pi = self.x = self.i_prev = self.eps_d = self.eps_s = self._t = None

    # ---------------------------------------------------------------- gym API #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pi = self.PI_STAR
        self.x = 0.0
        self.i_prev = self.R_N + self.PI_STAR
        self.eps_d = 0.0
        self.eps_s = 0.0
        self._t = 0
        if self.loader is not None:
            self.loader.reset(seed=seed)
        return self._observe(), {}

    def step(self, action):
        i_t = float(np.clip(np.asarray(action).item(), self.I_MIN, self.I_MAX))

        # Reward is evaluated on the state BEFORE transitioning (normalised).
        pi_dev = (self.pi - self.PI_STAR) / self.PI_SCALE
        x_dev = self.x / self.X_SCALE
        reward = -(pi_dev ** 2 + self.LAM * x_dev ** 2)

        eps_d_next = self.RHO_D * self.eps_d + self.rng.normal(0.0, self.SIGMA_D)
        eps_s_next = self.RHO_S * self.eps_s + self.rng.normal(0.0, self.SIGMA_S)

        Et_pi_next = self.OMEGA * self.pi + (1 - self.OMEGA) * self.PI_STAR
        x_next = (self.x - (1.0 / self.SIGMA) * (i_t - Et_pi_next - self.R_N)
                  + eps_d_next)
        pi_next = self.BETA * Et_pi_next + self.KAPPA * self.x + eps_s_next

        diverged = (abs(pi_next - self.PI_STAR) > self.PI_CLIP * 0.9
                    or abs(x_next) > self.X_CLIP * 0.9)
        if diverged:
            reward -= self.DIV_PENALTY

        pi_next = float(np.clip(pi_next, self.PI_STAR - self.PI_CLIP,
                                self.PI_STAR + self.PI_CLIP))
        x_next = float(np.clip(x_next, -self.X_CLIP, self.X_CLIP))

        self.pi, self.x, self.i_prev = pi_next, x_next, i_t
        self.eps_d, self.eps_s = eps_d_next, eps_s_next
        self._t += 1

        terminated = False
        truncated = self._t >= self.ep_len
        raw_reward = -((self.pi - self.PI_STAR) ** 2 + self.LAM * self.x ** 2)
        info = {"pi": self.pi, "x": self.x, "i": i_t,
                "reward": raw_reward, "diverged": diverged}
        return self._observe(), reward, terminated, truncated, info

    # ------------------------------------------------------------- internals #
    def _observe(self):
        if self.mode == "full_obs":
            return np.array([self.pi, self.x, self.i_prev], dtype=np.float32)
        belief = self.loader.get(self._t, self.pi, self.x)
        return np.array([belief[0], belief[1], belief[2], belief[3],
                         self.i_prev], dtype=np.float32)

    def taylor_action(self, clip: bool = False) -> float:
        """The analytic Taylor-rule rate, the benchmark policy."""
        i = (self.R_N + self.PI_STAR
             + self.PHI_PI * (self.pi - self.PI_STAR)
             + self.PHI_Y * self.x)
        return float(np.clip(i, self.I_MIN, self.I_MAX)) if clip else float(i)


# --------------------------------------------------------------------------- #
# Deterministic helpers (no Gym wrapper, no shocks) - used for validation.    #
# --------------------------------------------------------------------------- #
PI_STAR = NKEnv.PI_STAR
R_N = NKEnv.R_N
I_STAR = R_N + PI_STAR
BETA = NKEnv.BETA
SIGMA = NKEnv.SIGMA
KAPPA = NKEnv.KAPPA
PHI_PI = NKEnv.PHI_PI
PHI_Y = NKEnv.PHI_Y
OMEGA = NKEnv.OMEGA


def nk_step(pi: float, x: float, i_t: float) -> tuple[float, float]:
    """One deterministic step of the NK model (no shocks)."""
    et_pi = OMEGA * pi + (1 - OMEGA) * PI_STAR
    x_new = x - (1.0 / SIGMA) * (i_t - et_pi - R_N)
    pi_new = BETA * et_pi + KAPPA * x
    return pi_new, x_new


def taylor_rule(pi: float, x: float) -> float:
    """Analytic Taylor-rule rate."""
    return R_N + PI_STAR + PHI_PI * (pi - PI_STAR) + PHI_Y * x
