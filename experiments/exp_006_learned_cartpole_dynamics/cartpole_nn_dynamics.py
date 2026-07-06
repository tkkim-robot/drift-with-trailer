import jax.numpy as jnp
import optax
from flax import nnx

from src.learning.models.cartpole_data import Data


@nnx.jit
def loss_fn(model, batch):
    preds = model(batch[0])
    return ((preds - batch[1]) ** 2).mean()


@nnx.jit
def train_step(model, optimizer, metrics, batch):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
    metrics.update(loss=loss)

    return loss


# @nnx.jit
def eval_step(model, state):
    return model(state[None, ...])[0]


class LearnedDynamics:
    def __init__(
        self,
        model,
        batch_size,
        state_mean,
        state_std,
        dynamics_mean,
        dynamics_std,
        optimizer_params={"learning_rate": 0.05},
    ):
        self.model = model
        self.state_std = state_std
        self.state_mean = state_mean
        self.dynamics_std = dynamics_std
        self.dynamics_mean = dynamics_mean
        self.data = Data(batch_size, state_mean, state_std, dynamics_mean, dynamics_std)
        self.optimizer = nnx.Optimizer(self.model, optax.adam(**optimizer_params), wrt=nnx.Param)
        self.metrics = nnx.metrics.Average("loss")
        self.loss_history = []

    def __call__(self, state, action):
        full_state = jnp.concatenate([state, action])
        full_state = (full_state - self.state_mean) / self.state_std
        return self._unnormalize(self.model(full_state))
    

    def train(self, epochs):
        for i in range(epochs):
            for batch in zip(*self.data.get_data()):
                train_step(self.model, self.optimizer, self.metrics, batch)

            self.loss_history.append(self.metrics.compute())
            self.metrics.reset()

            print(f"Epoch {i}\tLoss: {self.loss_history[-1]}")

    def _unnormalize(self, dynamics):
        return dynamics * self.dynamics_std + self.dynamics_mean
