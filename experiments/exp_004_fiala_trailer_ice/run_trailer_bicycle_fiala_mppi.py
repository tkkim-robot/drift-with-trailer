from experiments.exp_004_fiala_trailer_ice.utils.trailer_driver import run_mpc

import jax.numpy as jnp

ctl_args = (
    jnp.diag(jnp.array([2e-3, 0.2])),
)
ctl_kwargs = {
    "inverse_temp": 1e-1,
    "K": 750,
    "step": 0.05,
    "T": 55,
    "alpha": 0.851356016887989,
}
cost_kwargs = {
    "reverse": False, 
    "v_target": -20,
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "s_weight": 1e2,
    "c_weight": 1e-2,
    "a_weight": 1e3,
}

if __name__ == "__main__":
        
    run_mpc(
        "MPPI",
        ctl_args,
        ctl_kwargs,
        cost_kwargs,
        record=True,
        debug=True,
        quiet=False,
        benchmark=False,
        headless=False,
        env_kwargs=None,
    )