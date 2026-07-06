"""The MS-GLSTM calibrated forecaster (TensorFlow/Keras).

Refactored verbatim from the ST456 notebook. TensorFlow is imported lazily so
the rest of the package and the test-suite import without it; call any function
here only in an environment with TensorFlow installed.

Model: a dual-stream LSTM encoder (monthly + quarterly frequencies) with a
heteroscedastic Gaussian head predicting (mu, log sigma^2) per target, trained
with a joint NLL + CRPS objective. See ``dl/README.md`` for provenance.
"""

from __future__ import annotations

import numpy as np

from dl.config import (
    MONTHLY_FEATURES, QUARTERLY_FEATURES, TARGET_COLS, SEQ_LEN, BATCH_SIZE,
    UNITS_MONTHLY, UNITS_QUARTERLY, UNITS_FUSED, LOG_VAR_CLAMP,
    LR, EPOCHS, PATIENCE, GRAD_CLIP_NORM, SIGMA_REG_WEIGHT, SEED, LAMBDA_GRID,
)
from dl.data import split_streams

try:  # TensorFlow is optional at import time.
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _HAS_TF = True
except Exception:  # pragma: no cover - exercised only without TF
    tf = keras = layers = None
    _HAS_TF = False


def _require_tf():
    if not _HAS_TF:
        raise ImportError(
            "TensorFlow is required for the MS-GLSTM. Install it with "
            "`pip install tensorflow` and re-run in that environment."
        )


# --------------------------------------------------------------------------- #
# Datasets                                                                     #
# --------------------------------------------------------------------------- #
def make_dataset(X: np.ndarray, y: np.ndarray, shuffle: bool = False):
    """Build a ((X_monthly, X_quarterly), y) tf.data pipeline."""
    _require_tf()
    X_m, X_q = split_streams(X)
    ds = tf.data.Dataset.from_tensor_slices(((X_m, X_q), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------------------------------- #
# Loss                                                                         #
# --------------------------------------------------------------------------- #
def gaussian_crps(y_true, mu, sigma):
    """Closed-form CRPS for a Gaussian predictive distribution.

    CRPS(N(mu, sigma^2), y) = sigma [ z(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ],
    with z = (y-mu)/sigma. A proper scoring rule (Gneiting & Raftery, 2007).
    """
    _require_tf()
    sigma = tf.maximum(sigma, 1e-6)
    z = (y_true - mu) / sigma
    Phi_z = 0.5 * (1.0 + tf.math.erf(z / tf.sqrt(2.0)))
    phi_z = tf.exp(-0.5 * z ** 2) / tf.sqrt(2.0 * np.pi)
    return tf.reduce_mean(
        sigma * (2.0 * phi_z + z * (2.0 * Phi_z - 1.0) - 1.0 / tf.sqrt(np.pi))
    )


def make_joint_loss(lam: float, reg_weight: float = SIGMA_REG_WEIGHT):
    """Loss = lam*NLL + (1-lam)*CRPS + reg_weight*mean(sigma^2).

    log sigma^2 is clamped inside the model to [-4, 4]; the sigma regulariser
    deters the head from gaming NLL with trivially small variance.
    """
    _require_tf()
    n_t = len(TARGET_COLS)

    def loss(y_true, y_pred):
        mu = y_pred[:, :n_t]
        log_var = y_pred[:, n_t:]
        sigma = tf.sqrt(tf.exp(log_var) + 1e-6)
        nll = tf.reduce_mean(
            0.5 * tf.reduce_sum(
                log_var + (y_true - mu) ** 2 / (sigma ** 2 + 1e-6), axis=-1)
        )
        crps = tf.reduce_mean(tf.stack([
            gaussian_crps(y_true[:, i], mu[:, i], sigma[:, i]) for i in range(n_t)
        ]))
        l_reg = tf.reduce_mean(tf.exp(log_var))
        return lam * nll + (1.0 - lam) * crps + reg_weight * l_reg

    return loss


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def build_msglstm(seq_len: int = SEQ_LEN,
                  n_monthly: int = len(MONTHLY_FEATURES),
                  n_quarterly: int = len(QUARTERLY_FEATURES),
                  n_targets: int = len(TARGET_COLS)):
    """Dual-stream LSTM encoder + heteroscedastic Gaussian head.

    Monthly:   LSTM(64, seq) -> LSTM(32) -> h_m
    Quarterly: LSTM(32, seq) -> LSTM(16) -> h_q
    Fusion:    concat -> Dense(48, tanh)
    Heads:     Dense(n_targets) for mu; Dense(n_targets) -> clip[-4,4] for log_var
    """
    _require_tf()
    inp_m = keras.Input(shape=(seq_len, n_monthly), name="monthly_input")
    h_m = layers.LSTM(UNITS_MONTHLY, return_sequences=True, name="lstm_m1")(inp_m)
    h_m = layers.LSTM(UNITS_MONTHLY // 2, name="lstm_m2")(h_m)

    inp_q = keras.Input(shape=(seq_len, n_quarterly), name="quarterly_input")
    h_q = layers.LSTM(UNITS_QUARTERLY, return_sequences=True, name="lstm_q1")(inp_q)
    h_q = layers.LSTM(UNITS_QUARTERLY // 2, name="lstm_q2")(h_q)

    h_fused = layers.Concatenate(name="concat")([h_m, h_q])
    h_fused = layers.Dense(UNITS_FUSED, activation="tanh", name="fusion")(h_fused)

    mu = layers.Dense(n_targets, name="mu")(h_fused)
    log_var_raw = layers.Dense(n_targets, name="log_var_raw")(h_fused)
    log_var = layers.Lambda(
        lambda x: tf.clip_by_value(x, -LOG_VAR_CLAMP, LOG_VAR_CLAMP),
        name="log_var")(log_var_raw)

    outputs = layers.Concatenate(name="output")([mu, log_var])
    return keras.Model(inputs=[inp_m, inp_q], outputs=outputs, name="msglstm")


# --------------------------------------------------------------------------- #
# Inference / selection / training                                            #
# --------------------------------------------------------------------------- #
def msglstm_predict(model, X: np.ndarray):
    """Return (mu, sigma) in normalised units, both shape (N, n_targets)."""
    _require_tf()
    X_m, X_q = split_streams(X)
    raw = model.predict([X_m, X_q], verbose=0)
    n_t = len(TARGET_COLS)
    mu = raw[:, :n_t]
    sigma = np.sqrt(np.exp(np.clip(raw[:, n_t:], -LOG_VAR_CLAMP, LOG_VAR_CLAMP)) + 1e-6)
    return mu, sigma


def val_crps(model, X: np.ndarray, y: np.ndarray) -> float:
    """Mean CRPS across targets on a validation set (the selection criterion)."""
    _require_tf()
    mu, sigma = msglstm_predict(model, X)
    n_t = len(TARGET_COLS)
    return float(np.mean([
        gaussian_crps(
            tf.constant(y[:, i], dtype=tf.float32),
            tf.constant(mu[:, i], dtype=tf.float32),
            tf.constant(sigma[:, i], dtype=tf.float32),
        ).numpy() for i in range(n_t)
    ]))


def train_msglstm_one(lam: float, splits: dict, verbose: int = 0):
    """Train one MS-GLSTM at a fixed lambda. Returns (model, history, val_crps, val_loss)."""
    _require_tf()
    keras.utils.set_random_seed(SEED)
    model = build_msglstm()
    model.compile(
        optimizer=keras.optimizers.Adam(LR, clipnorm=GRAD_CLIP_NORM),
        loss=make_joint_loss(lam),
    )
    ds_train = make_dataset(splits["train"]["X"], splits["train"]["y"], shuffle=True)
    ds_val = make_dataset(splits["val"]["X"], splits["val"]["y"])
    history = model.fit(
        ds_train, validation_data=ds_val, epochs=EPOCHS,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
        verbose=verbose,
    )
    vc = val_crps(model, splits["val"]["X"], splits["val"]["y"])
    vl = float(min(history.history["val_loss"]))
    return model, history, vc, vl


def lambda_grid_search(splits: dict, grid=LAMBDA_GRID):
    """Train one model per lambda; return (best_model, best_lam, results)."""
    _require_tf()
    results, best_lam, best_crps, best_model = {}, None, np.inf, None
    for lam in grid:
        model, history, vc, vl = train_msglstm_one(lam, splits)
        results[lam] = {"val_loss": vl, "val_crps": vc}
        if vc < best_crps:
            best_crps, best_lam, best_model = vc, lam, model
    return best_model, best_lam, results
