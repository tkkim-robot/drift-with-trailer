import jax.numpy as jnp
import jax

# Constants
POLE_LEN = 0.5
FORCE = 10

g = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = POLE_MASS + CART_MASS
POLEMASS_LENGTH = POLE_MASS * POLE_LEN



@jax.jit
def constraints(x, u):
    x_threshold = 2.4

    return ~((-x_threshold <= x[0]) & (x[0] <= x_threshold))


@jax.jit
def cost(x, u, t):
    return (
        x[2] ** 2 + x[0] ** 2 + 1**t * 10_000 * constraints(x, u).astype(jnp.int64)
    )  # + x[:, 3]**2 /10


@jax.jit
def term_cost(x, u):
    return x[3] ** 2 + x[1] ** 2


@jax.jit
def bound_control(u):
    return jnp.clip(u, -FORCE, FORCE)
