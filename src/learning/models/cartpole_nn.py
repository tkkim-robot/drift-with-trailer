from flax import nnx

class CartpoleModel(nnx.Module):
    def __init__(self, in_dim, out_dim):
        rng = nnx.Rngs(1248)

        self.model = nnx.Sequential(
            nnx.Linear(in_dim, 16, rngs=rng),
            nnx.tanh,
            nnx.Linear(16, 16, rngs=rng),
            nnx.tanh,
            nnx.Linear(16, out_dim, rngs=rng),
        )

    def __call__(self, x):
        return self.model(x) 
    