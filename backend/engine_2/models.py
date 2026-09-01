"""One definition of every network, imported by training, export and inference.

The notebook defined MultiHeadAttentionBlock three times (cell, CPU-export
subprocess string, and implicitly at load time) and — more seriously — trained
the PPO on `full_forecaster` outputs while exporting `forecaster` (the plain
BiLSTM baseline) to models/bilstm/model.json. The agent was therefore served a
different forecaster than the one it learned against. FORECASTER_NAME below is
the single switch that decides which model is trained on, backtested, and
shipped; nothing downstream is allowed to pick its own.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from . import config as C

FORECASTER_NAME = "cnn_bilstm_mha"      # or "bilstm"
HIDDEN_UNITS, DROPOUT, L2 = 128, 0.2, 1e-5
STATE_SIZE, ACTION_SIZE = 13, 3
COLLAPSE_MIN_STD, COLLAPSE_WEIGHT = 0.05, 20.0


@keras.utils.register_keras_serializable(package="custom")
class MultiHeadAttentionBlock(layers.Layer):
    def __init__(self, d_model, num_heads=4, dropout=0.1, **kw):
        super().__init__(**kw)
        self.d_model, self.num_heads, self.drop_rate = d_model, num_heads, dropout

    def build(self, input_shape):
        self.needs_projection = input_shape[-1] != self.d_model
        if self.needs_projection:
            self.input_proj = layers.Dense(self.d_model, name="mha_input_proj")
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.d_model // self.num_heads,
            dropout=self.drop_rate, name="mha_core")
        self.layernorm = layers.LayerNormalization(name="mha_layernorm")
        self.dropout = layers.Dropout(self.drop_rate)
        super().build(input_shape)

    def call(self, x, training=None):
        residual = self.input_proj(x) if self.needs_projection else x
        a = self.dropout(self.mha(query=x, value=x, key=x, training=training),
                         training=training)
        return self.layernorm(residual + a)

    def get_config(self):
        return super().get_config() | {"d_model": self.d_model,
                                       "num_heads": self.num_heads,
                                       "dropout": self.drop_rate}


@keras.utils.register_keras_serializable(package="custom")
def collapse_aware_bce(y_true, y_pred):
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
    avg_std = tf.reduce_mean(tf.math.reduce_std(y_pred, axis=0))
    return bce + tf.maximum(0.0, COLLAPSE_MIN_STD - avg_std) * COLLAPSE_WEIGHT


class DirectionalAccuracy(keras.metrics.Metric):
    def __init__(self, index=0, name="dir_acc", **kw):
        super().__init__(name=name, **kw)
        self.index = index
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        yt = tf.cast(y_true[:, self.index] > 0.5, tf.float32)
        yp = tf.cast(y_pred[:, self.index] > 0.5, tf.float32)
        m = tf.cast(tf.equal(yt, yp), tf.float32)
        self.total.assign_add(tf.reduce_sum(m))
        self.count.assign_add(tf.cast(tf.size(m), tf.float32))

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_state(self):
        self.total.assign(0.0); self.count.assign(0.0)


class BinaryAUCFromSoft(keras.metrics.AUC):
    def __init__(self, index=0, name="auc", **kw):
        super().__init__(name=name, **kw)
        self.index = index

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(tf.cast(y_true[:, self.index] > 0.5, tf.int32),
                                    y_pred[:, self.index], sample_weight)


def build_bilstm(seq_len=C.WINDOW_SIZE, num_features=C.NUM_FEATURES,
                 hidden=HIDDEN_UNITS, dropout=DROPOUT, horizon=C.HORIZON, l2_val=L2):
    reg = regularizers.L2(l2_val)
    inp = keras.Input(shape=(seq_len, num_features))
    x = layers.Bidirectional(layers.LSTM(hidden, return_sequences=True,
                                         kernel_regularizer=reg, recurrent_regularizer=reg),
                             name="bilstm_1")(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(hidden // 2, return_sequences=True,
                                         kernel_regularizer=reg, recurrent_regularizer=reg),
                             name="bilstm_2")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hidden, kernel_regularizer=reg,
                     bias_initializer=keras.initializers.Constant(0.01))(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon, activation="sigmoid", name="classifier_head")(x)
    return keras.Model(inp, out, name="BiLSTM")


def build_cnn_bilstm_mha(seq_len=C.WINDOW_SIZE, num_features=C.NUM_FEATURES,
                         hidden=HIDDEN_UNITS, dropout=DROPOUT,
                         horizon=C.HORIZON, l2_val=L2):
    reg = regularizers.L2(l2_val)
    inp = keras.Input(shape=(seq_len, num_features))
    branches = []
    for k in (3, 5, 7):
        b = layers.Conv1D(32, k, padding="same", activation="relu",
                          kernel_regularizer=reg, name=f"conv1d_k{k}")(inp)
        branches.append(layers.BatchNormalization(name=f"bn_k{k}")(b))
    x = layers.Concatenate()(branches)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(hidden, return_sequences=True,
                                         kernel_regularizer=reg, recurrent_regularizer=reg),
                             name="bilstm_1")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(hidden // 2, return_sequences=True,
                                         kernel_regularizer=reg, recurrent_regularizer=reg),
                             name="bilstm_2")(x)
    x = layers.Dropout(dropout)(x)
    x = MultiHeadAttentionBlock(hidden, 4, dropout, name="mha_block")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(hidden, kernel_regularizer=reg,
                     bias_initializer=keras.initializers.Constant(0.01))(x)
    x = layers.LeakyReLU(negative_slope=0.01)(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(horizon, activation="sigmoid", name="classifier_head")(x)
    return keras.Model(inp, out, name="CNN_BiLSTM_MHA")


def build_forecaster(name=FORECASTER_NAME, **kw):
    return {"bilstm": build_bilstm, "cnn_bilstm_mha": build_cnn_bilstm_mha}[name](**kw)


def build_actor(state_size=STATE_SIZE, action_size=ACTION_SIZE):
    inp = keras.Input(shape=(state_size,))
    x = layers.Dense(128, activation="tanh")(inp)
    x = layers.Dense(64, activation="tanh")(x)
    return keras.Model(inp, layers.Dense(action_size, activation="softmax")(x), name="Actor")


def build_critic(state_size=STATE_SIZE):
    inp = keras.Input(shape=(state_size,))
    x = layers.Dense(128, activation="tanh")(inp)
    x = layers.Dense(64, activation="tanh")(x)
    return keras.Model(inp, layers.Dense(1)(x), name="Critic")


CUSTOM_OBJECTS = {"MultiHeadAttentionBlock": MultiHeadAttentionBlock,
                  "collapse_aware_bce": collapse_aware_bce,
                  "DirectionalAccuracy": DirectionalAccuracy,
                  "BinaryAUCFromSoft": BinaryAUCFromSoft}


def load_bundle(models_dir=C.MODELS_DIR):
    load = lambda p: keras.models.load_model(p, custom_objects=CUSTOM_OBJECTS, compile=False)
    return (load(f"{models_dir}/forecaster/model.keras"),
            load(f"{models_dir}/ppo/policy/model.keras"),
            load(f"{models_dir}/ppo/value/model.keras"))


# ── state vector (13) — the ONLY definition ─────────────────────────────────
def assemble_state(candles, i, forecast, position, entry_price, bars_in_pos, volatility):
    o, high, low, close = candles[i, 1], candles[i, 2], candles[i, 3], candles[i, 4]
    pnl = (close - entry_price) / entry_price if position == 1 and entry_price > 0 else 0.0
    return np.array([
        1 - position, pnl, min(bars_in_pos / 100.0, 1.0),
        (o - close) / close, (high - close) / close, (low - close) / close,
        (high - low) / close, position, volatility, *forecast[:4],
    ], dtype=np.float32)


def make_policy(forecaster, actor, X, anchors, vols, candles, greedy=True, batch=1024):
    """Turn (forecaster, actor) into the callable the backtester expects.

    Forecasts are precomputed per window, then looked up by bar index — the live
    loop does the same computation one bar at a time, from the same features.
    """
    preds = np.concatenate([forecaster(X[i:i + batch], training=False).numpy()
                            for i in range(0, len(X), batch)], axis=0)
    by_bar = {int(a): k for k, a in enumerate(anchors)}

    def policy(i, s):
        k = by_bar.get(i)
        if k is None:
            return 0
        st = assemble_state(candles, i, preds[k], s["position"], s["entry_px"],
                            s["bars_in"], float(vols[k]))
        probs = actor(st[None, :], training=False).numpy()[0]
        return int(np.argmax(probs)) if greedy else \
            int(np.random.choice(len(probs), p=probs))

    policy.predictions = preds
    return policy
