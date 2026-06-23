from experiments.exp_004_fiala_trailer_ice.utils.trailer_driver import run_mpc

import jax.numpy as jnp

# Fwd Args (max 25)
# ctl_args = (
#     jnp.diag(jnp.array([4e-2, 0.2])),
#     jnp.diag(jnp.array([1e-2, 1e-1])),
# )
# ctl_kwargs = {
#     "inverse_temp": 0.25,
#     "K": 500,
#     "step": 0.05,
#     "T": 85,
#     "alpha": 0.05,
# }
# cost_kwargs = {
#     "reverse": False, 
#     "v_target": 25,
#     "p_weight": 1e2,
#     "p_slow_weight": 1e0,
#     "s_weight": 2e2,
#     "c_weight": 1e0,
#     "a_weight": 7e2,
# }

# Rev
ctl_args = (
    jnp.diag(jnp.array([3e-2, 0.2])),
    jnp.diag(jnp.array([1e-2, 1e-1])),
)
ctl_kwargs = {
    "inverse_temp": 0.5,
    "K": 750,
    "step": 0.05,
    "T": 55,
    "alpha": 0.05,
}
cost_kwargs = {
    "reverse": False, 
    "v_target": -25,
    "p_weight": 1e2,
    "p_slow_weight": 1e0,
    "s_weight": 1e2,
    "c_weight": 1e1,
    "a_weight": 1e2,
}

if __name__ == "__main__":
        
    run_mpc(
        "SMPPI",
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