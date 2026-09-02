"""Forecaster + PPO training, packaged as functions the scheduler and the
walk-forward harness can call without a human clicking through cells.

This is the notebook's logic with the corrections that mattered:

  * the model that trains the agent is the model that gets exported (models.py
    FORECASTER_NAME) — no baseline/full mix-up;
  * entries fill at the NEXT open, matching backtest.py, so the reward the agent
    maximizes is the return the backtest measures;
  * slippage scales with the volatility state feature, so the agent is not taught
    that a fill in a panic costs what a fill in a dead Sunday costs;
  * every parallel environment starts at its own random offset, so "512 envs"
    buys data diversity instead of 512 replays of one trajectory;
  * all four forecast horizons feed the reward through an agreement bonus, not
    just h1;
  * the hold-a-winner bonus decays to zero as unrealized gain approaches
    TAKE_PROFIT_PCT, so it reinforces the take-profit exit instead of competing
    with it;
  * PPO can warm-start from the previous champion at a reduced learning rate;
  * health checks are hard gates (gates.py) that raise — a collapsed forecaster
    stops the cycle before a single PPO update is spent on it.

    python -m engine_2.train                 # train on the current dataset.npz
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config as C
from . import gates
from . import models as M

# forecaster
LR, MIN_LR, EPOCHS, BATCH, PATIENCE = 5e-4, 1e-5, 60, 256, 10
# ppo
GAMMA, GAE_LAMBDA, CLIP_RATIO = 0.99, 0.95, 0.2
POLICY_LR, VALUE_LR, MIN_PPO_LR, LR_DECAY = 3e-4, 1e-3, 1e-5, 0.99
WARM_START_LR_SCALE = 0.25          # fine-tuning is a nudge, not a restart
MAX_GRAD_NORM = 0.5
PPO_EPOCHS, PPO_BATCH, NUM_PARALLEL = 10, 8192, 512
ROLLOUT_STEPS = 512                 # per env; with random offsets this covers far
                                    # more of the series than one long replay did
TOTAL_UPDATES, PPO_PATIENCE, FEE_WARMUP = 200, 10, 20
MIN_DELTA_FRAC = 0.0025

# entropy: start where the notebook did, but raise it automatically when the
# policy stops differentiating between states (the "always HOLD" failure), and
# decay it back once the policy is spread out again.
ENTROPY_COEFF = 0.01
ENTROPY_MAX, ENTROPY_MIN = 0.05, 0.002
ENTROPY_UP, ENTROPY_DOWN = 1.35, 0.97

# reward v3
PROFIT_BONUS, OPPORTUNITY_COST = 0.5, 0.3
HOLD_WINNER_BONUS, HOLD_LOSER_DECAY, LOSS_CUT_BONUS = 1.5, 0.25, 0.001
AGREEMENT_BONUS = 0.15              # paid on entry when all HORIZON heads agree
PRED_STRENGTH_TH = 0.05
LOSS_CUT_THRESHOLD = C.STOP_LOSS_PCT / 3.0


class CosineAnnealing(keras.callbacks.Callback):
    def __init__(self, lr_max, lr_min, total):
        super().__init__(); self.a, self.b, self.n = lr_max, lr_min, total

    def on_epoch_begin(self, epoch, logs=None):
        cos = 0.5 * (1 + math.cos(math.pi * epoch / self.n))
        self.model.optimizer.learning_rate.assign(self.b + (self.a - self.b) * cos)


# ── forecaster ───────────────────────────────────────────────────────────────
def train_forecaster(X_tr, y_tr, X_va, y_va, epochs=EPOCHS, verbose=1,
                     warm_start: str | None = None):
    if warm_start and os.path.exists(warm_start):
        model = keras.models.load_model(warm_start, custom_objects=M.CUSTOM_OBJECTS,
                                        compile=False)
        lr = LR * WARM_START_LR_SCALE
        print(f"  forecaster warm-started from {warm_start} (lr {lr:.2g})")
    else:
        model, lr = M.build_forecaster(), LR
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=M.collapse_aware_bce,
                  metrics=[M.DirectionalAccuracy(), M.BinaryAUCFromSoft()])
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
              batch_size=BATCH, shuffle=True, verbose=verbose,
              callbacks=[CosineAnnealing(lr, MIN_LR, epochs),
                         keras.callbacks.EarlyStopping(monitor="val_dir_acc", mode="max",
                                                       patience=PATIENCE, min_delta=1e-3,
                                                       restore_best_weights=True)])
    return model


def predict_all(model, X, batch=1024):
    return np.concatenate([model(X[i:i + batch], training=False).numpy()
                           for i in range(0, len(X), batch)], axis=0)


def forecaster_health(model, X, y) -> dict:
    """Every number the hard gate in gates.py judges, for all HORIZON heads."""
    p = predict_all(model, X)
    up = (y > 0.5).astype(np.int32)
    per_h = [{"h": h + 1,
              "pred_std": float(p[:, h].std()),
              "pred_mean": float(p[:, h].mean()),
              "dir_acc": float(((p[:, h] > 0.5).astype(np.int32) == up[:, h]).mean())}
             for h in range(p.shape[1])]
    agree = np.mean(np.abs(np.sign(p - 0.5).mean(axis=1)))
    return {"pred_mean": per_h[0]["pred_mean"], "pred_std": per_h[0]["pred_std"],
            "dir_acc": per_h[0]["dir_acc"], "up_rate": float(up[:, 0].mean()),
            "mean_dir_acc": float(np.mean([h["dir_acc"] for h in per_h])),
            "horizon_agreement": float(agree), "per_horizon": per_h}


# ── PPO ──────────────────────────────────────────────────────────────────────
def _ppo_graph(actor, critic, policy_opt, value_opt, entropy_coeff):
    @tf.function
    def rollout(opens, highs, lows, closes, forecasts, vols, offsets, n_steps, fee):
        """Each env walks its own slice of history, starting at offsets[e] and
        wrapping around the end of the series."""
        n_envs = tf.shape(offsets)[0]
        n_usable = tf.shape(forecasts)[0] - 1
        ta = lambda dt: tf.TensorArray(dt, size=n_steps)
        S, A, LP, V, R = ta(tf.float32), ta(tf.int32), ta(tf.float32), ta(tf.float32), ta(tf.float32)
        pos = tf.zeros([n_envs], tf.int32)
        entry = tf.zeros([n_envs], tf.float32)
        bars = tf.zeros([n_envs], tf.int32)

        for t in tf.range(n_steps):
            idx = tf.math.floormod(offsets + t, n_usable)            # [n_envs]
            nxt = idx + 1
            close = tf.gather(closes, idx)
            o, h, l = tf.gather(opens, idx), tf.gather(highs, idx), tf.gather(lows, idx)
            nxt_open, nxt_close = tf.gather(opens, nxt), tf.gather(closes, nxt)
            fc = tf.gather(forecasts, idx)                           # [n_envs, HORIZON]
            vol = tf.gather(vols, idx)

            safe = tf.where(entry > 0.0, entry, 1.0)
            pnl = tf.where((pos == 1) & (entry > 0.0), (close - safe) / safe, 0.0)
            agree = tf.reduce_mean(tf.sign(fc - 0.5), axis=1)         # [-1, 1]

            state = tf.concat([
                tf.stack([
                    tf.cast(1 - pos, tf.float32), pnl,
                    tf.minimum(tf.cast(bars, tf.float32) / 100.0, 1.0),
                    (o - close) / close, (h - close) / close, (l - close) / close,
                    (h - l) / close, tf.cast(pos, tf.float32), vol,
                ], axis=1),
                fc, agree[:, None],
            ], axis=1)

            probs = actor(state, training=False)
            values = tf.squeeze(critic(state, training=False), axis=1)
            u = tf.random.uniform(tf.shape(probs), 1e-5, 1 - 1e-5)
            act = tf.argmax(tf.math.log(probs + 1e-8) - tf.math.log(-tf.math.log(u)),
                            axis=-1, output_type=tf.int32)
            lp = tf.math.log(tf.gather_nd(probs, tf.stack([tf.range(n_envs), act], 1)) + 1e-8)

            forced = (pos == 1) & ((pnl <= -C.STOP_LOSS_PCT) | (pnl >= C.TAKE_PROFIT_PCT))
            eff = tf.where(forced, 2, act)

            # Costs are state-dependent: slippage rises with the same realized
            # volatility the policy can see, so it cannot learn to size its
            # aggression on a cost that only exists in calm markets.
            slip = C.SLIPPAGE_PCT * tf.clip_by_value(
                vol / C.REFERENCE_VOL, C.SLIPPAGE_VOL_MIN, C.SLIPPAGE_VOL_MAX)

            fill = nxt_open                     # decisions on close t, filled at open t+1
            price_ret = (nxt_close - close) / close
            strength = fc[:, 0] - 0.5

            flat, in_pos = pos == 0, pos == 1
            flat_buy, flat_hold = flat & (eff == 1), flat & (eff != 1)
            in_sell, in_hold = in_pos & (eff == 2), in_pos & (eff != 2)

            # entry: next-bar return net of costs, plus an edge bonus that is only
            # paid when the four horizons point the same way as the trade
            buy_r = price_ret - fee - slip
            buy_r = buy_r + tf.maximum(strength, 0.0) * OPPORTUNITY_COST
            buy_r = buy_r + tf.maximum(agree, 0.0) * AGREEMENT_BONUS * tf.abs(strength)
            hold_r = tf.where(strength > PRED_STRENGTH_TH,
                              -strength * OPPORTUNITY_COST * 0.3, 0.0)

            realized = tf.where(entry > 0.0,
                                (fill * (1 - slip)) / (safe * (1 + slip)) - 1.0, 0.0)
            sell_r = realized - 2.0 * fee
            sell_r = tf.where(realized > 0.0, sell_r + realized * PROFIT_BONUS, sell_r)
            sell_r = tf.where((realized < -LOSS_CUT_THRESHOLD) & (realized > -C.STOP_LOSS_PCT),
                              sell_r + LOSS_CUT_BONUS, sell_r)

            # Holding a winner: the bonus fades linearly to zero as unrealized
            # gain approaches TAKE_PROFIT_PCT. Uncapped, it paid the agent to ride
            # a position past the level the risk rules are about to close anyway,
            # which is how you teach a policy to fight its own exits.
            taper = tf.clip_by_value(1.0 - pnl / C.TAKE_PROFIT_PCT, 0.0, 1.0)
            time_f = tf.minimum(tf.cast(bars, tf.float32) / 20.0, 1.0)
            hold_pos_r = tf.where(
                pnl > 0.0,
                price_ret * (1.0 + (HOLD_WINNER_BONUS - 1.0) * taper),
                price_ret - tf.abs(pnl) * HOLD_LOSER_DECAY * time_f)

            rew = tf.zeros([n_envs], tf.float32)
            rew = tf.where(flat_buy, buy_r, rew)
            rew = tf.where(flat_hold, hold_r, rew)
            rew = tf.where(in_sell, sell_r, rew)
            rew = tf.where(in_hold, hold_pos_r, rew)

            pos = tf.where(flat_buy, 1, tf.where(in_sell, 0, pos))
            entry = tf.where(flat_buy, fill, tf.where(in_sell, 0.0, entry))
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
            loss = -tf.reduce_mean(tf.minimum(s1, s2)) - entropy_coeff * ent
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


def policy_spread(actor, states, sample=4096, seed=0) -> float:
    """How differently does the policy behave across unrelated market states?

    Mean over actions of the std of P(action | state). A policy that answers
    "36% HOLD / 28% BUY / 36% SELL" to everything scores ~0 here however good its
    average reward looks — that is the collapsed policy the notebook shipped, and
    it is a gate, not a print.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(states), size=min(sample, len(states)), replace=False)
    probs = actor(states[idx], training=False).numpy()
    return float(probs.std(axis=0).mean())


def train_ppo(candles, anchors, forecasts, vols, updates=TOTAL_UPDATES,
              verbose=True, warm_start_dir: str | None = None,
              n_envs=NUM_PARALLEL, rollout_steps=ROLLOUT_STEPS, seed=0):
    """-> (actor, critic, diagnostics). Warm start fine-tunes the previous
    champion at WARM_START_LR_SCALE of the usual learning rate."""
    warm = bool(warm_start_dir and
                os.path.exists(f"{warm_start_dir}/ppo/policy/model.keras"))
    if warm:
        load = lambda p: keras.models.load_model(p, custom_objects=M.CUSTOM_OBJECTS,
                                                 compile=False)
        actor = load(f"{warm_start_dir}/ppo/policy/model.keras")
        critic = load(f"{warm_start_dir}/ppo/value/model.keras")
        if actor.input_shape[-1] != M.STATE_SIZE:
            print(f"  warm start rejected: saved actor expects "
                  f"{actor.input_shape[-1]} state features, this build has "
                  f"{M.STATE_SIZE} — training from scratch")
            warm = False
    if not warm:
        actor, critic = M.build_actor(), M.build_critic()
    scale = WARM_START_LR_SCALE if warm else 1.0
    if warm and verbose:
        print(f"  PPO warm-started from {warm_start_dir} (lr x{scale})")

    p_opt = keras.optimizers.Adam(POLICY_LR * scale)
    v_opt = keras.optimizers.Adam(VALUE_LR * scale)
    p_opt.build(actor.trainable_variables); v_opt.build(critic.trainable_variables)
    ent_coeff = tf.Variable(ENTROPY_COEFF, dtype=tf.float32, trainable=False)
    rollout, upd_a, upd_c = _ppo_graph(actor, critic, p_opt, v_opt, ent_coeff)

    lo, hi = int(anchors[0]), int(anchors[-1]) + 2
    sl = candles[lo:hi]
    n_usable = len(forecasts) - 1
    steps = int(min(rollout_steps, max(32, n_usable)))
    args = [tf.constant(sl[:, k], tf.float32) for k in (1, 2, 3, 4)]
    args += [tf.constant(forecasts, tf.float32), tf.constant(vols, tf.float32)]

    rng = np.random.default_rng(seed)
    best, since_best, best_w = -np.inf, 0, None
    history = []
    for u in range(updates):
        t0 = time.time()
        fee = C.FEE_RATE * min(1.0, (u + 1) / FEE_WARMUP)
        # fresh random start offsets every update: the agent never sees the same
        # trajectory twice, so it cannot memorize one path through history
        offsets = tf.constant(rng.integers(0, max(1, n_usable), size=n_envs), tf.int32)
        st, ac, lp, va, rw = rollout(*args, offsets, tf.constant(steps),
                                     tf.constant(fee, tf.float32))
        st, ac, lp, va, rw = (x.numpy() for x in (st, ac, lp, va, rw))
        adv = _gae(rw, va)
        ret = adv + va
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        flat = lambda x: x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x.reshape(-1)
        fs, fa, flp, fadv, fret = map(flat, (st, ac, lp, adv, ret))
        idx = np.arange(len(fs))
        for _ in range(PPO_EPOCHS):
            rng.shuffle(idx)
            for s in range(0, len(idx), PPO_BATCH):
                b = idx[s:s + PPO_BATCH]
                upd_a(tf.constant(fs[b]), tf.constant(fa[b], tf.int32),
                      tf.constant(fadv[b]), tf.constant(flp[b]))
                upd_c(tf.constant(fs[b]), tf.constant(fret[b]))
        for opt, base in ((p_opt, POLICY_LR * scale), (v_opt, VALUE_LR * scale)):
            opt.learning_rate.assign(max(base * LR_DECAY ** (u + 1), MIN_PPO_LR))

        spread = policy_spread(actor, fs)
        # keep exploration alive exactly when the policy is going constant
        if spread < C.GATE_MIN_POLICY_SPREAD:
            ent_coeff.assign(min(ENTROPY_MAX, float(ent_coeff.numpy()) * ENTROPY_UP))
        else:
            ent_coeff.assign(max(ENTROPY_MIN, float(ent_coeff.numpy()) * ENTROPY_DOWN))

        avg = float(rw.sum(axis=0).mean())
        history.append({"update": u + 1, "avg_reward": avg, "policy_spread": spread,
                        "entropy_coeff": float(ent_coeff.numpy()), "fee": fee})
        warmup = u < FEE_WARMUP
        tag = " (warmup)" if warmup else ""
        if not warmup:
            if u == FEE_WARMUP or avg > best + max(abs(best), 1.0) * MIN_DELTA_FRAC:
                best, since_best, tag = avg, 0, " * best"
                best_w = (actor.get_weights(), critic.get_weights())
            else:
                since_best += 1
        if verbose:
            print(f"  upd {u+1:03d} fee={fee:.5f} avg_reward={avg:+.4f} "
                  f"spread={spread:.4f} ent={float(ent_coeff.numpy()):.4f} "
                  f"buys={float((ac==1).sum(0).mean()):.0f} "
                  f"sells={float((ac==2).sum(0).mean()):.0f} "
                  f"{time.time()-t0:.0f}s{tag}")
        if not warmup and since_best >= PPO_PATIENCE:
            print(f"  early stop at update {u+1}; best avg_reward {best:+.4f}")
            break

    if best_w:
        actor.set_weights(best_w[0]); critic.set_weights(best_w[1])
    diag = {"best_avg_reward": best if np.isfinite(best) else None,
            "final_policy_spread": policy_spread(actor, fs),
            "warm_started": warm, "updates_run": len(history),
            "n_envs": n_envs, "rollout_steps": steps, "history": history[-20:]}
    return actor, critic, diag


# ── one full training pass over a prepared slice bundle ──────────────────────
def train_bundle(d: dict, forecaster_epochs=EPOCHS, ppo_updates=TOTAL_UPDATES,
                 warm_start_dir: str | None = None, verbose=True,
                 enforce_gates: bool = True) -> dict:
    """dataset dict -> {forecaster, actor, critic, health, ppo}. Raises GateFailed
    the moment the forecaster is not good enough to be worth an agent."""
    fc = train_forecaster(d["X_train"], d["y_train"], d["X_val"], d["y_val"],
                          epochs=forecaster_epochs, verbose=2 if verbose else 0,
                          warm_start=(f"{warm_start_dir}/forecaster/model.keras"
                                      if warm_start_dir else None))
    health = forecaster_health(fc, d["X_val"], d["y_val"])
    if verbose:
        print(f"  forecaster val: dir_acc={health['dir_acc']:.4f} "
              f"predStd={health['pred_std']:.4f} predMean={health['pred_mean']:.3f} "
              f"meanDirAcc={health['mean_dir_acc']:.4f}")
    if enforce_gates:
        gates.check_forecaster(health)          # raises before any PPO compute

    preds = predict_all(fc, d["X_train"])
    actor, critic, ppo_diag = train_ppo(d["candles"], d["anchor_train"], preds,
                                        d["vol_train"], updates=ppo_updates,
                                        verbose=verbose, warm_start_dir=warm_start_dir)
    if enforce_gates:
        gates.check_policy(ppo_diag)
    return {"forecaster": fc, "actor": actor, "critic": critic,
            "health": health, "ppo": ppo_diag}


def train_fold(fold: dict, forecaster_epochs=EPOCHS, ppo_updates=TOTAL_UPDATES,
               verbose=True):
    """walk-forward entry point: fold dict -> policy callable.

    A fold that fails the gates is not an error — it is a fold where the model
    would not have been deployed, so it trades nothing and its flat curve counts
    against the consistency check. That is the honest accounting.
    """
    d = {"X_train": fold["X_train"], "y_train": fold["y_train"],
         "X_val": fold["X_val"], "y_val": fold["y_val"],
         "candles": fold["candles"], "anchor_train": fold["anchors_train"],
         "vol_train": fold["vol_train"]}
    try:
        out = train_bundle(d, forecaster_epochs, ppo_updates, verbose=verbose)
    except gates.GateFailed as exc:
        print(f"  fold gated out ({exc}) -> flat policy, no trades")
        return lambda i, s: 0
    return M.make_policy(out["forecaster"], out["actor"], fold["X_test"],
                         fold["anchors_test"], fold["vol_test"], fold["candles"])


def save_bundle(out_dir: str, forecaster, actor, critic, feat_mean, feat_std,
                meta: dict | None = None):
    import json
    for sub, m in (("forecaster", forecaster), ("ppo/policy", actor), ("ppo/value", critic)):
        os.makedirs(f"{out_dir}/{sub}", exist_ok=True)
        m.save(f"{out_dir}/{sub}/model.keras")
    np.savez(f"{out_dir}/scaler.npz", mean=feat_mean, std=feat_std)
    with open(f"{out_dir}/meta.json", "w") as fh:
        json.dump(meta or {}, fh, indent=2, default=float)
    return out_dir


if __name__ == "__main__":
    from .dataset import load

    d = load()
    out = train_bundle(d)
    save_bundle(C.CANDIDATE_DIR, out["forecaster"], out["actor"], out["critic"],
                d["feat_mean"], d["feat_std"],
                {"health": out["health"], "ppo": out["ppo"], "meta": d["meta"]})
    print(f"candidate models -> {C.CANDIDATE_DIR}")
