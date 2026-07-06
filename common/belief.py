"""Belief-state loaders: the DL -> RL bridge.

The RL agent conducts monetary policy under partial observability. Instead of
seeing the true macro state (pi, x) it sees a *belief* about it:

    [mu_pi, sigma2_pi, mu_x, sigma2_x]

i.e. a mean and variance for inflation and the output gap. Where that belief
comes from is the seam between the deep-learning forecaster and the RL agent.

Three loaders live here:

* ``BeliefStateLoader`` - the abstract contract (mirrors the original ST455 env).
* ``PlaceholderLoader``  - the ORIGINAL stub: near-true means, fixed sigma. Kept
  verbatim as the baseline the assessor flagged ("nearly true state means and
  fixed uncertainty magnitudes").
* ``ForecasterBeliefLoader`` - the Tier-1 replacement that fixes both flaws.

--------------------------------------------------------------------------------
What the Tier-1 loader changes, and why it stays honest
--------------------------------------------------------------------------------
The placeholder conflated two different quantities. We separate them:

* ``tau`` - the *true* forecast-error scale (reality). A forecaster's mean is
  not the true state; it is the true state plus error of size ~tau.
* ``sigma`` - the uncertainty the forecaster *reports* to the agent.

Calibration is precisely whether the reported ``sigma`` matches the real error
``tau``:

* calibrated:    sigma ~= tau        (honest; intervals cover at their nominal rate)
* overconfident: sigma <  tau        (reported intervals are too tight -> under-cover)

So both conditions now receive genuinely NOISY means (drawn with the same tau,
which keeps the experiment a clean one-variable contrast), and they differ only
in whether the reported variance tells the truth. That is the actual definition
of calibration, and it sharpens - rather than replaces - the original
calibrated-vs-overconfident design.

We also make the uncertainty TIME-VARYING: tau and sigma scale with a local
volatility estimate, tuned so that turbulent periods widen the interval ~1.2x.
That number is not invented - it is the MS-GLSTM's *measured* behaviour: its
predicted sigma rose ~1.22x during the COVID shock in the ST456 evaluation.
Optionally, the overconfident loader's sigma widens LESS with volatility, so
overconfidence gets worse exactly when reliable uncertainty matters most - the
phenomenon the ST456 report identified for the transformer backbone.

This is a *forecaster-informed, calibration-faithful* belief model: its sigma
magnitudes come from the real DL evaluation and its dynamics reproduce the real
forecaster's observed behaviour. It does NOT run the live MS-GLSTM inside the
simulation - that is Tier 3 (see README), and is deliberately out of scope here
because the DL model is trained on real FRED history while this environment is a
synthetic New-Keynesian simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Calibration constants from the ST456 (DL) pre-COVID evaluation.             #
# Reproduced from the ST455 notebook so the bridge is self-contained.         #
#   normalised sigma * target-std (* 100 for inflation -> annualised %).      #
# --------------------------------------------------------------------------- #
_PI_STD = 0.03635
_X_STD = 1.73551
_PCT = 100.0

# MS-GLSTM (well-calibrated): reports large, honest uncertainty.
SIGMA_PI_CALIB = 0.6201 * _PI_STD * _PCT   # ~2.254% p.a.
SIGMA_X_CALIB = 0.5305 * _X_STD            # ~0.920 pp

# MC Dropout (overconfident): reports small uncertainty that under-covers.
SIGMA_PI_OVERC = 0.1255 * _PI_STD * _PCT   # ~0.456% p.a.
SIGMA_X_OVERC = 0.0906 * _X_STD            # ~0.157 pp

# Steady-state / scaling references (match NKEnv).
PI_STAR = 2.0
PI_SCALE = 1.0
X_SCALE = 2.0

# Observed MS-GLSTM sigma widening during the COVID shock (ST456 evaluation).
COVID_SIGMA_WIDENING = 1.22


# --------------------------------------------------------------------------- #
# Contract                                                                     #
# --------------------------------------------------------------------------- #
class BeliefStateLoader(ABC):
    """Return a belief [mu_pi, sigma2_pi, mu_x, sigma2_x] as float32."""

    @abstractmethod
    def get(self, t: int, pi_true: float, x_true: float) -> np.ndarray: ...

    def reset(self, seed=None) -> None:  # optional per-episode reset
        pass


# --------------------------------------------------------------------------- #
# Original stub (baseline the assessor flagged) - kept verbatim               #
# --------------------------------------------------------------------------- #
class PlaceholderLoader(BeliefStateLoader):
    """ORIGINAL ST455 loader: near-true means + fixed sigma.

    Retained unchanged so the merged repo can reproduce the original results and
    quantify what the Tier-1 belief model changes.
    """

    def __init__(self, sigma_pi, sigma_x, noise_scale=0.01, seed=42):
        self.sigma_pi = sigma_pi
        self.sigma_x = sigma_x
        self.noise = noise_scale
        self._base_seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        s = seed if seed is not None else self._base_seed
        self.rng = np.random.default_rng(s)

    def get(self, t, pi_true, x_true):
        mu_pi = pi_true + self.rng.normal(0, self.noise)
        mu_x = x_true + self.rng.normal(0, self.noise)
        return np.array([mu_pi, self.sigma_pi ** 2,
                         mu_x, self.sigma_x ** 2], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Tier-1 forecaster-informed belief model                                      #
# --------------------------------------------------------------------------- #
@dataclass
class BeliefConfig:
    """Configuration for :class:`ForecasterBeliefLoader`.

    tau_*   : true forecast-error std (the noise actually injected into means).
    sigma_* : reported uncertainty std the agent observes.
              calibrated  -> set sigma == tau
              overconfident -> set sigma <  tau
    vol_*   : local-volatility response of the uncertainty (time variation).
    sigma_vol_ratio : how much the *reported* sigma tracks volatility relative to
              tau. 1.0 => sigma widens with turbulence just like the true error
              (calibrated stays calibrated). <1.0 => reported sigma under-widens
              in turbulent periods (overconfidence worsens in crises).
    """
    tau_pi: float
    tau_x: float
    sigma_pi: float
    sigma_x: float
    vol_gain: float = 0.15          # turbulence -> widening slope
    vol_cap: float = 1.6            # max widening multiplier
    vol_alpha: float = 0.35         # EWMA weight on new turbulence
    sigma_vol_ratio: float = 1.0    # reported-sigma volatility tracking (<=1)
    seed: int = 42

    @classmethod
    def calibrated(cls, **overrides) -> "BeliefConfig":
        """MS-GLSTM-style: reported sigma == true error; fully volatility-tracking."""
        cfg = dict(tau_pi=SIGMA_PI_CALIB, tau_x=SIGMA_X_CALIB,
                   sigma_pi=SIGMA_PI_CALIB, sigma_x=SIGMA_X_CALIB,
                   sigma_vol_ratio=1.0)
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def overconfident(cls, **overrides) -> "BeliefConfig":
        """MC-Dropout-style: true error is the honest (calibrated) scale, but the
        agent is *told* the smaller MC-Dropout sigma, which also under-widens in
        turbulence."""
        cfg = dict(tau_pi=SIGMA_PI_CALIB, tau_x=SIGMA_X_CALIB,
                   sigma_pi=SIGMA_PI_OVERC, sigma_x=SIGMA_X_OVERC,
                   sigma_vol_ratio=0.5)
        cfg.update(overrides)
        return cls(**cfg)


class ForecasterBeliefLoader(BeliefStateLoader):
    """Tier-1 belief loader: noisy means + time-varying, calibration-aware sigma.

    Drop-in replacement for :class:`PlaceholderLoader` (same ``get`` signature),
    so the RL environment needs no change beyond which loader it is handed.
    """

    def __init__(self, config: BeliefConfig):
        self.cfg = config
        self._base_seed = config.seed
        self.rng = np.random.default_rng(config.seed)
        self._ewma_turb = 0.0  # smoothed local turbulence

    def reset(self, seed=None):
        s = seed if seed is not None else self._base_seed
        self.rng = np.random.default_rng(s)
        self._ewma_turb = 0.0

    # -- local volatility ---------------------------------------------------- #
    def _volatility_multiplier(self, pi_true: float, x_true: float) -> float:
        """EWMA of state turbulence -> a widening multiplier in [1, vol_cap].

        Turbulence is distance of the state from target in scaled units; it is
        ~0 at the steady state and grows under large shocks / policy errors.
        """
        turb = 0.5 * abs(pi_true - PI_STAR) / PI_SCALE + 0.5 * abs(x_true) / X_SCALE
        self._ewma_turb = ((1 - self.cfg.vol_alpha) * self._ewma_turb
                           + self.cfg.vol_alpha * turb)
        mult = 1.0 + self.cfg.vol_gain * self._ewma_turb
        return float(np.clip(mult, 1.0, self.cfg.vol_cap))

    def get(self, t, pi_true, x_true):
        v = self._volatility_multiplier(pi_true, x_true)

        # True error scale widens fully with volatility.
        tau_pi_t = self.cfg.tau_pi * v
        tau_x_t = self.cfg.tau_x * v

        # Reported sigma tracks volatility only up to sigma_vol_ratio.
        v_rep = 1.0 + self.cfg.sigma_vol_ratio * (v - 1.0)
        sigma_pi_t = self.cfg.sigma_pi * v_rep
        sigma_x_t = self.cfg.sigma_x * v_rep

        # Genuinely noisy means, drawn with the TRUE error scale.
        mu_pi = pi_true + self.rng.normal(0.0, tau_pi_t)
        mu_x = x_true + self.rng.normal(0.0, tau_x_t)

        return np.array([mu_pi, sigma_pi_t ** 2,
                         mu_x, sigma_x_t ** 2], dtype=np.float32)


def make_loader(kind: str, **overrides) -> BeliefStateLoader:
    """Convenience factory: 'calibrated' | 'overconfident' | 'placeholder_calib'
    | 'placeholder_overconf'."""
    kind = kind.lower()
    if kind == "calibrated":
        return ForecasterBeliefLoader(BeliefConfig.calibrated(**overrides))
    if kind == "overconfident":
        return ForecasterBeliefLoader(BeliefConfig.overconfident(**overrides))
    if kind == "placeholder_calib":
        return PlaceholderLoader(SIGMA_PI_CALIB, SIGMA_X_CALIB, **overrides)
    if kind == "placeholder_overconf":
        return PlaceholderLoader(SIGMA_PI_OVERC, SIGMA_X_OVERC, **overrides)
    raise ValueError(f"Unknown loader kind {kind!r}.")
