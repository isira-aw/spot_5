"""Forecaster + PPO training, packaged as a function so walk-forward can call it
K times without a human clicking through cells.

This is the notebook's logic with three corrections:
  * the model that trains the agent is the model that gets exported (models.py
    FORECASTER_NAME) — no baseline/full mix-up;
  * entries fill at the NEXT open, matching backtest.py, so the reward the agent
    maximizes is the return the backtest measures;
  * early stopping and checkpoint selection watch validation directional
    accuracy, and a collapsed forecaster (predStd below the noise floor) aborts
    the fold instead of quietly training an agent on a constant.

Run standalone to train on the current dataset.npz:
    python -m trader.train
"""
from __future__ import annotations

import math
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config as C
from . import models as M

# forecaster
LR, MIN_LR, EPOCHS, BATCH, PATIENCE = 5e-4, 1e-5, 60, 256, 10
# ppo
GAMMA, GAE_LAMBDA, CLIP_RATIO = 0.99, 0.95, 0.2
POLICY_LR, VALUE_LR, MIN_PPO_LR, LR_DECAY = 3e-4, 1e-3, 1e-5, 0.99
ENTROPY_COEFF, MAX_GRAD_NORM = 0.01, 0.5
PPO_EPOCHS, PPO_BATCH, NUM_PARALLEL = 10, 8192, 256
TOTAL_UPDATES, PPO_PATIENCE, FEE_WARMUP = 200, 10, 20
MIN_DELTA_FRAC = 0.0025
# reward v2
PROFIT_BONUS, OPPORTUNITY_COST = 0.5, 0.3
HOLD_WINNER_BONUS, HOLD_LOSER_DECAY, LOSS_CUT_BONUS = 1.5, 0.25, 0.001
PRED_STRENGTH_TH = 0.05
LOSS_CUT_THRESHOLD = C.STOP_LOSS_PCT / 3.0
DEGENERATE_STD = 1e-4


class CosineAnnealing(keras.callbacks.Callback):
    def __init__(self, lr_max, lr_min, total):
        super().__init__(); self.a, self.b, self.n = lr_max, lr_min, total

    def on_epoch_begin(self, epoch, logs=None):
        cos = 0.5 * (1 + math.cos(math.pi * epoch / self.n))
        self.model.optimizer.learning_rate.assign(self.b + (self.a - self.b) * cos)


def train_forecaster(X_tr, y_tr, X_va, y_va, epochs=EPOCHS, verbose=1):
    model = M.build_forecaster()
    model.compile(optimizer=keras.optimizers.Adam(LR), loss=M.collapse_aware_bce,
                  metrics=[M.DirectionalAccuracy(), M.BinaryAUCFromSoft()])
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
              batch_size=BATCH, shuffle=True, verbose=verbose,
              callbacks=[CosineAnnealing(LR, MIN_LR, epochs),
                         keras.callbacks.EarlyStopping(monitor="val_dir_acc", mode="max",
                                                       patience=PATIENCE, min_delta=1e-3,
                                                       restore_best_weights=True)])
    return model


def forecaster_health(model, X, y):
    p = np.concatenate([model(X[i:i + 1024], training=False).numpy()
                        for i in range(0, len(X), 1024)], axis=0)[:, 0]
    up = (y[:, 0] > 0.5).astype(np.int32)
    return {"pred_mean": float(p.mean()), "pred_std": float(p.std()),
            "dir_acc": float(((p > 0.5).astype(np.int32) == up).mean()),
            "up_rate": float(up.mean())}


# ── PPO ──────────────────────────────────────────────────────────────────────
def _ppo_graph(actor, critic, policy_opt, value_opt):
    @tf.function
    def rollout(opens, highs, lows, closes, forecasts, vols, n_envs, n_steps, fee):
        ta = lambda dt: tf.TensorArray(dt, size=n_steps)
        S, A, LP, V, R = ta(tf.float32), ta(tf.int32), ta(tf.float32), ta(tf.float32), ta(tf.float32)
        pos = tf.zeros([n_envs], tf.int32)
        entry = tf.zeros([n_envs], tf.float32)
        bars = tf.zeros([n_envs], tf.int32)

        for t in tf.range(n_steps):
            close, o, h, l = closes[t], opens[t], highs[t], lows[t]
            nxt_open = opens[t + 1]
            fc = forecasts[t]
            cb = tf.fill([n_envs], close)
            safe = tf.where(entry > 0.0, entry, 1.0)
            pnl = tf.where((pos == 1) & (entry > 0.0), (cb - safe) / safe, 0.0)

            state = tf.stack([
                tf.cast(1 - pos, tf.float32), pnl,
                tf.minimum(tf.cast(bars, tf.float32) / 100.0, 1.0),
                tf.fill([n_envs], (o - close) / close),
                tf.fill([n_envs], (h - close) / close),
                tf.fill([n_envs], (l - close) / close),
                tf.fill([n_envs], (h - l) / close),
                tf.cast(pos, tf.float32), tf.fill([n_envs], vols[t]),
            ], axis=1)
            state = tf.concat([state, tf.tile(fc[None, :4], [n_envs, 1])], axis=1)

            probs = actor(state, training=False)
            values = tf.squeeze(critic(state, training=False), axis=1)
            u = tf.random.uniform(tf.shape(probs), 1e-5, 1 - 1e-5)
            act = tf.argmax(tf.math.log(probs + 1e-8) - tf.math.log(-tf.math.log(u)),
                            axis=-1, output_type=tf.int32)
            lp = tf.math.log(tf.gather_nd(probs, tf.stack([tf.range(n_envs), act], 1)) + 1e-8)

            forced = (pos == 1) & ((pnl <= -C.STOP_LOSS_PCT) | (pnl >= C.TAKE_PROFIT_PCT))
            eff = tf.where(forced, 2, act)

            # fills happen at the NEXT open, so that is the price the reward uses
            fill = nxt_open
            price_ret = (closes[t + 1] - close) / close
            strength = fc[0] - 0.5

            flat, in_pos = pos == 0, pos == 1
            flat_buy, flat_hold = flat & (eff == 1), flat & (eff != 1)
            in_sell, in_hold = in_pos & (eff == 2), in_pos & (eff != 2)

            buy_r = price_ret - fee - C.SLIPPAGE_PCT
            buy_r = buy_r + tf.maximum(strength, 0.0) * OPPORTUNITY_COST
            hold_r = tf.where(strength > PRED_STRENGTH_TH,
                              -strength * OPPORTUNITY_COST * 0.3, 0.0)
            realized = tf.where(entry > 0.0, (fill * (1 - C.SLIPPAGE_PCT)) /
                                (safe * (1 + C.SLIPPAGE_PCT)) - 1.0, 0.0)
            sell_r = realized - 2.0 * fee
            sell_r = tf.where(realized > 0.0, sell_r + realized * PROFIT_BONUS, sell_r)
            sell_r = tf.where((realized < -LOSS_CUT_THRESHOLD) & (realized > -C.STOP_LOSS_PCT),
                              sell_r + LOSS_CUT_BONUS, sell_r)
            tf_ = tf.minimum(tf.cast(bars, tf.float32) / 20.0, 1.0)
            hold_pos_r = tf.where(pnl > 0.0, price_ret * HOLD_WINNER_BONUS,
                                  price_ret - tf.abs(pnl) * HOLD_LOSER_DECAY * tf_)

            rew = tf.zeros([n_envs], tf.float32)
            rew = tf.where(flat_buy, tf.fill([n_envs], 0.0) + buy_r, rew)
            rew = tf.where(flat_hold, tf.fill([n_envs], 0.0) + hold_r, rew)
            rew = tf.where(in_sell, sell_r, rew)
            rew = tf.where(in_hold, tf.fill([n_envs], 0.0) + hold_pos_r, rew)

            pos = tf.where(flat_buy, 1, tf.where(in_sell, 0, pos))
            entry = tf.where(flat_buy, tf.fill([n_envs], fill),
                             tf.where(in_sell, 0.0, entry))
            bars = tf.where(flat_buy | in_sell, 0, tf.where(in_hold, bars + 1, bars))

            S, A, LP = S.write(t, state), A.write(t, act), LP.write(t, lp)
            V, R = V.write(t, values), R.write(t, rew)

        return S.stack(), A.stack(), LP.stack(), V.stack(), R.stack()

    @tf.function
    def upd_actor(st, ac, adv, old_lp):
        with tf.GradientTape() as tape:
            probs = actor(st, training=True)
            logp = tf.math.log(probs + 1e-8)
            idx = tf.stack([tf.range(tf.shape(ac)[0]), ac], axis=1)
            ratio = tf.exp(tf.gather_nd(logp, idx) - old_lp)
            s1 = ratio * adv
            s2 = tf.clip_by_value(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * adv
            ent = -tf.reduce_mean(tf.reduce_sum(probs * logp, axis=1))
            loss = -tf.reduce_mean(tf.minimum(s1, s2)) - ENTROPY_COEFF * ent
        g = [tf.clip_by_norm(x, MAX_GRAD_NORM)
             for x in tape.gradient(loss, actor.trainable_variables)]
        policy_opt.apply_gradients(zip(g, actor.trainable_variables))
        return loss, ent

    @tf.function
    def upd_critic(st, ret):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(ret - tf.squeeze(critic(st, training=True), 1)))
        g = [tf.clip_by_norm(x, MAX_GRAD_NORM)
             for x in tape.gradient(loss, critic.trainable_variables)]
        value_opt.apply_gradients(zip(g, critic.trainable_variables))
        return loss

    return rollout, upd_actor, upd_critic


def _gae(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], np.float32)
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(rewards) else np.zeros_like(last)
        mask = 0.0 if t == len(rewards) - 1 else 1.0
        delta = rewards[t] + gamma * nxt * mask - values[t]
        last = delta + gamma * lam * mask * last
        adv[t] = last
    return adv


def train_ppo(candles, anchors, forecasts, vols, updates=TOTAL_UPDATES, verbose=True):
    actor, critic = M.build_actor(), M.build_critic()
    p_opt, v_opt = keras.optimizers.Adam(POLICY_LR), keras.optimizers.Adam(VALUE_LR)
    p_opt.build(actor.trainable_variables); v_opt.build(critic.trainable_variables)
    rollout, upd_a, upd_c = _ppo_graph(actor, critic, p_opt, v_opt)

    lo, hi = int(anchors[0]), int(anchors[-1]) + 2
    sl = candles[lo:hi]
    n_steps = len(forecasts) - 1
    args = [tf.constant(sl[:, k], tf.float32) for k in (1, 2, 3, 4)]
    args += [tf.constant(forecasts, tf.float32), tf.constant(vols, tf.float32)]

    best, since_best, best_w = -np.inf, 0, None
    for u in range(updates):
        t0 = time.time()
        fee = C.FEE_RATE * min(1.0, (u + 1) / FEE_WARMUP)
        st, ac, lp, va, rw = rollout(*args, tf.constant(NUM_PARALLEL), 
                                     tf.constant(n_steps), tf.constant(fee, tf.float32))
        st, ac, lp, va, rw = (x.numpy() for x in (st, ac, lp, va, rw))
        adv = _gae(rw, va)
        ret = adv + va
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        flat = lambda x: x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x.reshape(-1)
        fs, fa, flp, fadv, fret = map(flat, (st, ac, lp, adv, ret))
        idx = np.arange(len(fs))
        for _ in range(PPO_EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, len(idx), PPO_BATCH):
                b = idx[s:s + PPO_BATCH]
                upd_a(tf.constant(fs[b]), tf.constant(fa[b], tf.int32),
                      tf.constant(fadv[b]), tf.constant(flp[b]))
                upd_c(tf.constant(fs[b]), tf.constant(fret[b]))
        for opt, base in ((p_opt, POLICY_LR), (v_opt, VALUE_LR)):
            opt.learning_rate.assign(max(base * LR_DECAY ** (u + 1), MIN_PPO_LR))

        avg = float(rw.sum(axis=0).mean())
        warm = u < FEE_WARMUP
        tag = " (warmup)" if warm else ""
        if not warm:
            if u == FEE_WARMUP or avg > best + max(abs(best), 1.0) * MIN_DELTA_FRAC:
                best, since_best, tag = avg, 0, " * best"
                best_w = (actor.get_weights(), critic.get_weights())
            else:
                since_best += 1
        if verbose:
            print(f"  upd {u+1:03d} fee={fee:.5f} avg_reward={avg:+.4f} "
                  f"buys={float((ac==1).sum(0).mean()):.0f} "
                  f"sells={float((ac==2).sum(0).mean()):.0f} "
                  f"{time.time()-t0:.0f}s{tag}")
        if not warm and since_best >= PPO_PATIENCE:
            print(f"  early stop at update {u+1}; best avg_reward {best:+.4f}")
            break

    if best_w:
        actor.set_weights(best_w[0]); critic.set_weights(best_w[1])
    return actor, critic


def train_fold(fold: dict, forecaster_epochs=EPOCHS, ppo_updates=TOTAL_UPDATES,
               verbose=True):
    """walk-forward entry point: fold dict -> policy callable."""
    fc = train_forecaster(fold["X_train"], fold["y_train"],
                          fold["X_val"], fold["y_val"],
                          epochs=forecaster_epochs, verbose=2 if verbose else 0)
    h = forecaster_health(fc, fold["X_val"], fold["y_val"])
    print(f"  forecaster val: dir_acc={h['dir_acc']:.4f} predStd={h['pred_std']:.4f} "
          f"predMean={h['pred_mean']:.3f}")
    if h["pred_std"] < DEGENERATE_STD:
        print("  forecaster collapsed -> flat policy for this fold (no trades)")
        return lambda i, s: 0

    preds = np.concatenate([fc(fold["X_train"][i:i + 1024], training=False).numpy()
                            for i in range(0, len(fold["X_train"]), 1024)], axis=0)
    actor, _ = train_ppo(fold["candles"], fold["anchors_train"], preds,
                         fold["vol_train"], updates=ppo_updates, verbose=verbose)
    return M.make_policy(fc, actor, fold["X_test"], fold["anchors_test"],
                         fold["vol_test"], fold["candles"])


if __name__ == "__main__":
    import os
    from .dataset import load

    d = load()
    fc = train_forecaster(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    print(forecaster_health(fc, d["X_val"], d["y_val"]))
    preds = np.concatenate([fc(d["X_train"][i:i + 1024], training=False).numpy()
                            for i in range(0, len(d["X_train"]), 1024)], axis=0)
    actor, critic = train_ppo(d["candles"], d["anchor_train"], preds, d["vol_train"])
    for sub, m in (("forecaster", fc), ("ppo/policy", actor), ("ppo/value", critic)):
        os.makedirs(f"{C.CANDIDATE_DIR}/{sub}", exist_ok=True)
        m.save(f"{C.CANDIDATE_DIR}/{sub}/model.keras")
    np.savez(f"{C.CANDIDATE_DIR}/scaler.npz",
             mean=d["feat_mean"], std=d["feat_std"])
    print(f"candidate models -> {C.CANDIDATE_DIR}")
