from experiments.exp_004_fiala_trailer_ice.utils.bicycle_driver import run_mpc

import jax.numpy as jnp

# Configs to be used

mppi_cfg = (
    (
    jnp.diag(jnp.array([0.01, 0.1])),
    ),
    {
        "inverse_temp": 0.5,
        "K": 500,
        "step": 0.05,
        "T": 85,
        "alpha": 0.05,
    },
    {
        "reverse": False, 
        "v_target": -30,
        "p_weight": 1e1,
        "p_slow_weight": 1e0,
        "s_weight": 2e1,
        "c_weight": 2e1,
    },
)

smppi_cfg = (
    (
        jnp.diag(jnp.array([0.01, 0.1])),
        jnp.diag(jnp.array([1e-1, 1e-1])),
    ),
    {
        "inverse_temp": 0.5,
        "K": 500,
        "step": 0.05,
        "T": 85,
        "alpha": 0.05,
    },
    {
        "reverse": False, 
        "v_target": 30,
        "p_weight": 1e1,
        "p_slow_weight": 1e0,
        "s_weight": 2e1,
        "c_weight": 3e1,
    },
)

cfg = [mppi_cfg, smppi_cfg]


if __name__ == "__main__":


    trials = [
        ("MPPI For.; no v bound", "SMPPI For.;no v bound", False, None),
        ("MPPI Rev.; no v bound", "SMPPI Rev.;no v bound", True, None),
        ("MPPI For.;  v_t = 90 ", "SMPPI For.; v_t = 90 ", False, 90),
        ("MPPI For.;  v_t = 120", "SMPPI For.; v_t = 120", False, 120),
        ("MPPI Rev.;  v_t = 90 ", "SMPPI Rev.; v_t = 90 ", False, -90),
    ]

    for config_mppi, config_smppi, reverse, v_t in trials:
        
        name = [config_mppi, config_smppi]

        for i in range(2):

            v_tt = None if v_t is None else v_t / 3.6
            
            ctl_args, ctl_kwargs, cost_kwargs = cfg[i]
            cost_kwargs["reverse"] = reverse
            cost_kwargs["v_target"] = v_tt

            mode = "MPPI" if i == 0 else "SMPPI"

            run_mpc(
                mode,
                ctl_args,
                ctl_kwargs,
                cost_kwargs,
                record=True,
                benchmark=True,
                quiet=True,
                debug=True,
                max_steps=3000,
                print_name=name[i],
                record_file_name=f"{mode}_v={str(v_tt)}_r={reverse}"
            )
        