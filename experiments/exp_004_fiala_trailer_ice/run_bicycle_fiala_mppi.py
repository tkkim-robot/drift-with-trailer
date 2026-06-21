from experiments.exp_004_fiala_trailer_ice.utils.bicycle_driver import run_mpc

import jax.numpy as jnp

ctl_args = (
    jnp.diag(jnp.array([1e-3, 0.06115174214186951])),
)
ctl_kwargs = {
    "inverse_temp": 0.1,
    "K": 750,
    "step": 0.05,
    "T": 75,
    "alpha": 0.851356016887989,
}
cost_kwargs = {
    "reverse": False, 
    "v_target": -30,
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "s_weight": 808.4087416389013,
    "c_weight": 0.012476149239058718,
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