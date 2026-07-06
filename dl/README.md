# `dl/` — the calibrated forecaster (ST456)

This package is the deep-learning half of the project: the **MS-GLSTM**, a
dual-stream LSTM with a heteroscedastic Gaussian head trained on a joint
NLL + CRPS objective to produce *calibrated* probabilistic forecasts of
inflation and the (negated) unemployment gap.

## Scope of this module

The full ST456 study — the complete model zoo (ARIMA, MC Dropout, Deep
Ensembles, the ARIMA-LSTM hybrid, Temporal Transformers), the ablations, and
the FRED download / feature engineering — lives in the original notebook at
`notebooks/ST456_DL_calibrated_forecasting.ipynb`, which was run on Colab and is
the authoritative artifact.

This package extracts the pieces that matter for the merged pipeline:

- `config.py` — architecture, training, split, and feature constants.
- `data.py` — sequence construction (TensorFlow-free and tested). Consumes the
  cached `master_df.csv` / `norm_params.json` the notebook produces.
- `msglstm.py` — the model, the CRPS/NLL loss, and train / predict routines
  (TensorFlow is imported lazily; call these in a TF environment).
- `calibration.py` — the **DL → RL handoff**: recomputes, from saved
  predictions, the per-regime uncertainty magnitudes that
  `common/belief.py` uses to parameterise the RL agent's belief state.

## Provenance, honestly

The forecaster's weights and prediction files are produced on Colab (with a
FRED API key) and are **not** redistributed here; `artifacts/dl/` is git-ignored.
The code in this module is a faithful refactor of the notebook, but — unlike the
RL half — it is not re-run in CI, because it requires TensorFlow and live FRED
data. The tested surface is the TF-free data and calibration logic. Treat the
notebook as the validated source of the DL numbers, and this module as the clean,
importable form of the forecaster.

## How it connects to the RL side

`common/belief.py` is parameterised by the calibrated (MS-GLSTM) and
overconfident (MC Dropout) σ magnitudes measured here. `calibration.py`
documents those constants and can regenerate them from a predictions `npz`, so
the bridge's numbers are reproducible rather than hard-coded magic. See the
top-level `README.md` for the full narrative.
