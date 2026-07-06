import jax.numpy as jnp
import jax


class Data:
    def __init__(self, batch_size, state_mean, state_std, dynamics_mean, dynamics_std):
        self.states = []
        self.dynamics = []
        self.state_std = state_std
        self.state_mean = state_mean
        self.dynamics_std = dynamics_std
        self.dynamics_mean = dynamics_mean
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(1248)

    def __len__(self):
        return len(self.states)

    def add(self, state, dynamics):
        state, dynamics = self._normalize(state, dynamics)

        self.states.append(state)
        self.dynamics.append(dynamics)

    def get_data(self):
        self.key, subkey = jax.random.split(self.key)

        states = jnp.array(self.states)
        dynamics = jnp.array(self.dynamics)

        n = states.shape[0]
        leftover = n % self.batch_size
        perm = jax.random.permutation(subkey, n)[leftover:]

        batched_states = states[perm].reshape(-1, self.batch_size, *states.shape[1:])
        batched_dynamics = dynamics[perm].reshape(-1, self.batch_size, *dynamics.shape[1:])

        return batched_states, batched_dynamics
    
    def _normalize(self, states, dynamics):
        states = (states - self.state_mean) / self.state_std
        dynamics = (dynamics - self.dynamics_mean) / self.dynamics_std

        return states, dynamics