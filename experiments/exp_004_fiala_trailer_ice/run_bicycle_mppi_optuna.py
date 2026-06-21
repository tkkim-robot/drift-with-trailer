from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import optuna

from src.controllers.mpc.mppi_jax import MPPI_Jax
from src.dynamics.vehicle.bicycle_fiala import gen_util_funs
from src.simulation.bicycle_env import BicycleEnv


@dataclass
class Config:
    K: int = 1000  
    T: int = 100 
    step: float = 0.05  
    max_steps: int = 2500
    n_starts: int = 2 
    start_speed: float = 8.0 
    reverse: bool = False
    v_target_kmh: float | None = 150.0 
    p_weight_anchor: float = 1e2 
    p_slow_weight: float = 1.0
    mppi_alpha: float = 0.05
    report_every: int = 150  
    crash_score: float = -1.0e3         # NaN/inf blow-up sentinel (meters-scale)



def _rollout_once(mpc, env, track, veh, cfg: Config,
                  start_progress: float, seed: int,
                  trial: optuna.Trial | None, report_offset: int):
    # Reset controller internal state so starts are independent, but DO reuse the
    # same MPPI instance so the jitted functions are not recompiled per start.
    mpc.last_trajectory = None
    mpc.key = jax.random.key(seed)

    env.reset(options={"initial_progress": float(start_progress),
                       "initial_speed": float(cfg.start_speed)})
    env.step(jnp.array([0.0, 0.0]))  # populate env._last_index for the arc seed

    direction = -1.0 if cfg.reverse else 1.0
    progress = 0.0
    prev_arc = None
    last_idx = env.unwrapped._last_index
    speeds, abs_slip_r, abs_yaw = [], [], []
    crashed = False
    steps = 0

    for i in range(cfg.max_steps):
        st = env.unwrapped._state
        if not np.all(np.isfinite([st.x, st.y, st.vx, st.vy, st.yaw_rate])):
            crashed = True
            break

        mu_here = float(env.unwrapped.track.find_mu(st.x, st.y))
        arc_seed = float(env.unwrapped.track._arc_samples[env.unwrapped._last_index])
        mpc_state = jnp.array([st.x, st.y, st.yaw, st.vx, st.vy, st.yaw_rate,
                               mu_here, arc_seed])

        u = mpc.run_mpc(mpc_state)
        u.block_until_ready()
        u = np.asarray(u)
        if not np.all(np.isfinite(u)):
            crashed = True
            break

        _, _, terminated, _, _ = env.step(jnp.array([float(u[0]), float(u[1])]))
        steps = i + 1

        st2 = env.unwrapped._state
        proj, last_idx = track.project(float(st2.x), float(st2.y), last_idx)
        arc = float(proj.arc_length)
        if prev_arc is not None:
            raw = arc - prev_arc
            d = raw - track.length * round(raw / track.length)  # unwrap across S/F line
            progress += direction * d
        prev_arc = arc

        speeds.append(float(np.hypot(st2.vx, st2.vy)))
        vx_safe = max(abs(float(st2.vx)), 0.5)
        abs_slip_r.append(abs(float(-np.arctan2(st2.vy - veh.lr * st2.yaw_rate, vx_safe))))
        abs_yaw.append(abs(float(st2.yaw_rate)))

        if trial is not None and (i % cfg.report_every == 0):
            trial.report(report_offset + max(progress, 0.0), step=report_offset + i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if terminated:  # out-of-bounds: normal early stop, progress stands as the penalty
            break

    stats = {
        "progress_m": progress,
        "steps": steps,
        "laps": progress / track.length,
        "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
        "mean_abs_slip_r": float(np.mean(abs_slip_r)) if abs_slip_r else 0.0,
        "max_abs_slip_r": float(np.max(abs_slip_r)) if abs_slip_r else 0.0,
        "mean_abs_yaw": float(np.mean(abs_yaw)) if abs_yaw else 0.0,
        "crashed": crashed,
    }
    score = cfg.crash_score if crashed else progress
    return score, stats


# TODO remove alpha param its useless
def make_objective(cfg: Config):
    env = BicycleEnv(renderer=None, render_mode=None)  # headless: no pybullet/cv2
    params = env.unwrapped.scenario
    veh = params.vehicle
    track = env.unwrapped.track
    v_target = None if cfg.v_target_kmh is None else cfg.v_target_kmh / 3.6

    def objective(trial: optuna.Trial) -> float:
        # ---- search space (6-D, all meaningful) ----
        lam = trial.suggest_float("lambda", 1e-2, 1e3, log=True)
        alpha_is = trial.suggest_float("alpha_is", 0.0, 1.0)        # gamma = (1-alpha)*lambda
        cv_steer = trial.suggest_float("cv_steer", 1e-3, 1.0, log=True)
        cv_accel = trial.suggest_float("cv_accel", 1e-2, 4.0, log=True)
        s_weight = trial.suggest_float("s_weight", 1e2, 1e9, log=True)
        c_weight = trial.suggest_float("c_weight", 1e-2, 1e3, log=True)

        gamma = (1.0 - alpha_is) * lam
        trial.set_user_attr("gamma_derived", gamma)

        dynamics, cost, bound, _ = gen_util_funs(
            params,
            reverse=cfg.reverse,
            v_target=v_target,
            p_weight=cfg.p_weight_anchor,
            p_slow_weight=cfg.p_slow_weight,
            s_weight=s_weight,
            c_weight=c_weight,
        )

        mpc = MPPI_Jax(
            6, 2,
            dynamics, None, cost, bound,
            jnp.diag(jnp.array([cv_steer, cv_accel])),
            inverse_temp=lam,
            alpha=cfg.mppi_alpha,
            gamma=gamma,
            K=cfg.K,
            step=cfg.step,
            T=cfg.T,
        )

        scores, agg = [], {k: [] for k in
                           ("progress_m", "mean_speed", "mean_abs_slip_r",
                            "max_abs_slip_r", "mean_abs_yaw", "laps", "steps")}
        any_crash = False
        for k in range(cfg.n_starts):
            start_progress = (k / max(cfg.n_starts, 1)) % 1.0
            score, stats = _rollout_once(
                mpc, env, track, veh, cfg,
                start_progress=start_progress, seed=k,
                trial=trial, report_offset=k * cfg.max_steps,
            )
            scores.append(score)
            any_crash = any_crash or stats["crashed"]
            for key in agg:
                agg[key].append(stats[key])

        for key, vals in agg.items():
            trial.set_user_attr(key, float(np.mean(vals)))
        trial.set_user_attr("any_crash", any_crash)

        return float(np.mean(scores))

    return objective



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--study-name", default="mppi_car_task004")
    ap.add_argument("--storage", default="sqlite:///optuna_mppi_car.db")
    ap.add_argument("--K", type=int, default=Config.K)
    ap.add_argument("--T", type=int, default=Config.T)
    ap.add_argument("--max-steps", type=int, default=Config.max_steps)
    ap.add_argument("--n-starts", type=int, default=Config.n_starts)
    ap.add_argument("--v-target-kmh", type=float, default=Config.v_target_kmh,
                    help="target speed in km/h; pass a negative number for None (max-speed mode)")
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 trial, tiny budget, sanity check")
    args = ap.parse_args()

    cfg = Config(
        K=args.K, T=args.T, max_steps=args.max_steps, n_starts=args.n_starts,
        reverse=args.reverse,
        v_target_kmh=(None if args.v_target_kmh is not None and args.v_target_kmh < 0
                      else args.v_target_kmh),
    )
    if args.smoke:
        cfg.max_steps, cfg.n_starts, args.trials = 80, 1, 1

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=cfg.max_steps // 3),
    )

    study.optimize(make_objective(cfg), n_trials=args.trials, gc_after_trial=True)

    print("\n=== best trial ===")
    bt = study.best_trial
    print(f"value (progress, m): {bt.value:.2f}")
    print("params:", json.dumps(bt.params, indent=2))
    print("derived gamma:", bt.user_attrs.get("gamma_derived"))
    print("stats:", json.dumps(
        {k: bt.user_attrs.get(k) for k in
         ("mean_speed", "laps", "mean_abs_slip_r", "max_abs_slip_r", "any_crash")},
        indent=2))

    with open("best_mppi_car_params.json", "w") as f:
        json.dump({"value": bt.value, "params": bt.params,
                   "gamma_derived": bt.user_attrs.get("gamma_derived")}, f, indent=2)
    print("\nwrote best_mppi_car_params.json")


if __name__ == "__main__":
    main()