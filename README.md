# Drift With Trailer

Research code for building from cartpole MPC baselines toward vehicle/trailer dynamics, friction variation, data collection, learned dynamics, and RMA-style adaptation.

## Repository Layout

- `docs/reading-list/`: papers, notes, and background material.
- `docs/meeting-notes/`: weekly advising and project meeting notes.
- `docs/tasks/`: student-facing assignments and check-in tasks.
- `src/`: reusable project code.
- `configs/`: vehicle, controller, and experiment configuration files.
- `examples/`: runnable examples and command-line entrypoints.
- `experiments/`: experiment-specific code and results organization.
- `tests/`: automated tests.
- `results/`: shared figures and tables.

## Python Setup With uv

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the local environment:

```bash
uv sync
```

Run the cartpole smoke simulation:

```bash
uv run python examples/cartpole_smoke.py
```

The smoke simulation is intentionally simple. It only checks that the Python environment, dependency installation, and a basic cartpole dynamics example are working.

## First Student Task

Start with [Assignment 001: Nonlinear MPC CartPole Swing-Up](docs/tasks/001_nonlinear_mpc_cartpole_swingup.md).
