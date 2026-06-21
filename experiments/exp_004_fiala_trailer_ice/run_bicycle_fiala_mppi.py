from experiments.exp_004_fiala_trailer_ice.utils.bicycle_driver import run_mpc

import jax.numpy as jnp

ctl_args = (
    jnp.diag(jnp.array([0.01, 0.1])),
)
ctl_kwargs = {
    "inverse_temp": 0.5,
    "K": 500,
    "step": 0.05,
    "T": 85,
    "alpha": 0.05,
}
cost_kwargs = {
    "reverse": False, 
    "v_target": 30,
    "p_weight": 1e1,
    "p_slow_weight": 1e0,
    "s_weight": 2e1,
    "c_weight": 2e1,
}

if __name__ == "__main__":
        
    run_mpc(
        "MPPI",
        ctl_args,
        ctl_kwargs,
        cost_kwargs,
        record=False,
        debug=True,
        quiet=False,
        benchmark=False,
        headless=False,
        env_kwargs=None,
    )