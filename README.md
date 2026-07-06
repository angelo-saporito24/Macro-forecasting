# Calibrated Forecasts and Monetary Policy under Uncertainty

An end-to-end pipeline linking two questions that are usually studied apart:

1. **Can a deep model forecast the macroeconomy with *honest* uncertainty?**
   A multi-scale LSTM (MS-GLSTM) trained with a CRPS objective produces
   calibrated probabilistic forecasts of inflation and the output gap.
2. **Does that honesty matter for policy?** A reinforcement-learning central
   bank sets interest rates in a New-Keynesian economy under partial
   observability, seeing only a *belief* about the state — and we ask whether
   the **calibration** of that belief changes the policy it learns.

The two halves meet at one interface: the forecaster's calibrated uncertainty
becomes the belief state the RL agent conditions on.

> **Origin.** This began as two LSE group projects — ST456 (deep learning) and
> ST455 (reinforcement learning) — designed as a matched pair. This repository
> merges them into one pipeline and adds the extension described below. See
> [Attribution](#attribution).

---

## The headline result

The RL project originally fed the agent a *placeholder* belief: near-true state
means with fixed uncertainty. Its assessor named exactly this as the main
limitation — the agent never interacts with genuinely uncertain, time-varying
forecasts. This repo closes that gap with a **forecaster-informed belief model**
(`common/belief.py`): genuinely noisy means, and time-varying σ whose magnitude
and crisis-widening come from the MS-GLSTM's *measured* behaviour. Calibration
is operationalised properly — as whether the *reported* σ matches the *true*
forecast-error scale.

The full study runs {PPO, SAC} × {placeholder, forecaster-informed} beliefs ×
3 seeds. Two findings emerge, stated at their true strength.

**1. Realistic beliefs collapse the policy's inflation response (robust).** The
POMDP agent's long-run Taylor coefficient φ_π falls from ~1.0–1.3 under
placeholder beliefs to ~0.02 under forecaster-informed beliefs — in *every seed,
under both PPO and SAC*. Once beliefs carry realistic, time-varying uncertainty,
the agent effectively abandons the Taylor principle: a structural loss of policy
credibility even where in-sample loss stays low.

Long-run φ_π (mean over seeds; Taylor-rule benchmark = 1.66):

| beliefs | PPO calibrated | PPO overconf. | SAC calibrated | SAC overconf. |
|---|--:|--:|--:|--:|
| placeholder | 1.05 | 1.01 | 1.27 | 1.31 |
| forecaster-informed | **0.018** | **0.019** | **0.020** | **0.016** |

**2. Calibration's *welfare* effect is modest and interacts with the algorithm.**
The calibrated-vs-overconfident reward gap is small everywhere (≈0.1–1.4 loss
units) and its sign depends on the setup: under PPO it flips (overconfident
marginally better with placeholder beliefs, calibrated better with realistic
ones); under SAC calibrated is marginally ahead in both. So calibration helps
*modestly*, not uniformly — it interacts with the RL algorithm rather than
operating in isolation. It should not be read as a decisive "calibration wins."

Welfare (mean episode reward, higher = better):

| beliefs | PPO calib. | PPO overc. | SAC calib. | SAC overc. |
|---|--:|--:|--:|--:|
| placeholder | −6.45 | **−6.04** | **−9.53** | −9.63 |
| forecaster-informed | **−14.85** | −16.28 | **−15.89** | −16.21 |

**Notes.** Numbers are the full-budget run (10⁶ steps × 3 seeds), cached in
`artifacts/rl/`. Full-observability agents are unaffected by the belief model
(they see the true state) and serve as a control; under PPO one seed's full-obs
policy is unstable and diverges in evaluation, so full-obs is reported at its
median. SAC did not diverge on any seed.

---

## Relationship to the assessor's feedback

The ST455 submission scored 78/100. The assessor praised two things and named
one limitation, and this repo speaks to all three:

- *"a policy can achieve a relatively low central-bank loss while still
  responding too weakly to inflation"* — finding 1 is exactly this, now shown
  structurally (φ_π → ~0) under realistic beliefs.
- *"the effect of uncertainty calibration interacts with the choice of RL
  algorithm, rather than operating in isolation"* — finding 2, now backed by
  matched PPO/SAC results.
- *"the RL agents do not interact directly with live time-varying forecasts
  during training: the belief-state loader uses nearly true state means and
  fixed uncertainty magnitudes"* — the `ForecasterBeliefLoader` replaces that
  stub with noisy means and time-varying, calibration-aware σ.

---

## How the two halves connect

The RL environment observes the state through a `BeliefStateLoader`:

```python
class BeliefStateLoader:
    def get(self, t, pi_true, x_true) -> np.ndarray:  # [mu_pi, sigma2_pi, mu_x, sigma2_x]
```

Three implementations, in increasing fidelity:

- `PlaceholderLoader` — the original stub (near-true means, fixed σ). Kept so the
  baseline is reproducible.
- `ForecasterBeliefLoader` — **this repo's contribution**: noisy means +
  time-varying, calibration-aware σ, parameterised by the forecaster's measured
  uncertainty (via `dl/calibration.py`).
- *(future — Tier 3)* the live MS-GLSTM inside the simulation. Deliberately out
  of scope: the forecaster is trained on real FRED history while the RL
  environment is a synthetic NK model, so the honest intermediate step is a
  forecaster-*informed* belief model, not the model itself.

---

## Repository layout

```
macro-forecasting-rl/
├── common/belief.py       # the DL->RL bridge + the Tier-1 belief model
├── rl/
│   ├── env.py             # New-Keynesian POMDP (Gymnasium)
│   ├── train.py           # PPO / SAC training (both belief modes)
│   ├── evaluate.py        # deterministic eval + Taylor-projection regression
│   └── run.py             # CLI driver over the condition matrix
├── dl/
│   ├── config.py          # forecaster constants
│   ├── data.py            # sequence building (TF-free, tested)
│   ├── msglstm.py         # the MS-GLSTM model + CRPS/NLL loss (TF, lazy import)
│   ├── calibration.py     # DL->RL handoff: sigma provenance
│   └── README.md          # scope + provenance of the DL half
├── notebooks/             # the two original ST455 / ST456 notebooks (authoritative)
├── artifacts/             # cached results (rl/) and DL outputs (dl/, git-ignored)
└── tests/                 # 25 tests; run without TensorFlow or any API key
```

---

## Quickstart

```bash
pip install -r requirements.txt   # TensorFlow NOT required for the RL half
pytest -q                         # 25 tests, no keys, no downloads
```

Reproduce the headline comparison (reduced budget, ~5 min/mode on CPU):

```bash
python -m rl.run --algos ppo --belief tier1 placeholder --seeds 42 \
    --steps-ppo 100000 --eval-episodes 20
```

Full reproduction as in the report (heavy — use a GPU/Colab):

```bash
python -m rl.run --algos ppo sac --belief tier1 placeholder --seeds 42 123 7
```

Fast end-to-end sanity check:

```bash
python -m rl.run --smoke
```

The **DL forecaster** (`dl/`) requires TensorFlow and a FRED API key; its
weights and predictions are produced on Colab and cached. See `dl/README.md`.

---

## What each part demonstrates

- **RL under partial observability** — a custom Gymnasium NK environment, PPO and
  SAC agents (stable-baselines3), belief-state observations.
- **Uncertainty quantification, applied** — calibration is not just measured on
  the forecaster; it is propagated into a downstream decision problem and shown
  to change behaviour.
- **Econometric evaluation** — a Taylor-principle projection (HC3-robust OLS)
  distinguishes in-sample welfare from structural policy credibility.
- **Reproducible engineering** — modular, seeded, cached artifacts, 25 tests, CI.

---

## Attribution

These were two four-person LSE group projects. **Angelo Saporito** proposed both
projects, proposed and designed their merger, and led the overall direction; the
merged pipeline and the forecaster-informed belief-model extension in this
repository are his own follow-on work.

The original notebooks are preserved unmodified under `notebooks/` as the record
of the group work.

## License

MIT — see `LICENSE`.
