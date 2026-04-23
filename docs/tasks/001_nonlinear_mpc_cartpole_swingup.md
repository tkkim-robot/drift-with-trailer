# Task 001: Nonlinear MPC CartPole Swing-Up With IPOPT

## Purpose

This is the first check-in task for the repository. The goal is to implement a nonlinear model predictive controller for the cartpole swing-up problem using IPOPT as the nonlinear optimization solver.

This assignment is intentionally scoped to cartpole. Do not implement the car or trailer dynamics yet. We will use this task to practice the GitHub workflow, code organization, experiment documentation, and controller tuning process that will later support vehicle/trailer MPC, sampling-based MPC, learned dynamics, friction variation, data collection, and RMA-style adaptation.

## Learning Goals

By the end of this task, you should be able to:

- Set up the Python environment with `uv`.
- Create a GitHub issue for your work.
- Create a branch, commit changes, and open a pull request.
- Implement a reusable nonlinear MPC controller.
- Use IPOPT through a Python optimization interface such as CasADi.
- Tune MPC horizon length, costs, constraints, and solver settings for cartpole swing-up.
- __(add this answer to the PR)__ Explain why a simple PID controller is usually not enough for this swing-up setting.

## Repository Locations

Put (reusable) controller code here:

```text
src/controllers/mpc/ipopt_cartpole.py
```

Put experiment-specific code here:

```text
experiments/exp_001_mpc_baseline/
```

Put runnable entrypoints here:

```text
examples/
```

For this check-in task, the cartpole dynamics may live inside the experiment folder. You do not need to create reusable dynamics code under `src/dynamics/` yet.

Good examples of experiment-specific files:

```text
experiments/exp_001_mpc_baseline/cartpole_dynamics.py
experiments/exp_001_mpc_baseline/plot_results.py
(optional) experiments/exp_001_mpc_baseline/config.yaml
experiments/exp_001_mpc_baseline/README.md
```

Good examples of runnable entrypoints:

```text
examples/run_cartpole_mpc.py
```

## GitHub Workflow

1. Create a GitHub issue titled:

```text
Task 001: Nonlinear MPC CartPole Swing-Up
```

2. In the issue, briefly describe your plan:

- What dynamics model you will use.
- What optimization library you will use.
- What files you expect to edit.
- What result will count as success.

3. Create a branch from `main`. Use a branch name like:

```text
task-001-cartpole-mpc
```

4. Make small commits with clear messages. Example:

```text
Add cartpole dynamics for MPC experiment
Add IPOPT-based nonlinear MPC controller
Tune cartpole swing-up costs and constraints
Document assignment 001 results
```

5. Open a pull request. Use this PR description template:

```markdown
## Summary

- Implemented nonlinear MPC for cartpole swing-up using IPOPT.
- Added runnable entrypoint under `examples/`.
- Added experiment files under `experiments/exp_001_mpc_baseline/`.
- Added reusable MPC controller under `src/controllers/mpc/`.

## Results

- Initial condition:
- Final cart position:
- Final pole angle:
- Maximum force used:
- Solver success rate:

## Tuning Notes

- Horizon length:
- Time step:
- State cost:
- Control cost:
- Terminal cost:
- Important constraints:

## Answers

### Why PID is not enough for cartpole swing-up in this setting?

### What might need to change to let PID also complete this task?

## Checklist

- [ ] Code runs with `uv run ...`
- [ ] Controller code is in `src/controllers/mpc/`
- [ ] Experiment code is in `experiments/exp_001_mpc_baseline/`
- [ ] Runnable entrypoint is in `examples/`
- [ ] IPOPT is used as the solver
- [ ] Results are described in the PR
- [ ] The two written questions are answered
```

## Technical Requirements

Use nonlinear MPC, not linear MPC. The optimization problem should plan over a finite horizon and solve for a sequence of cart forces.

Your implementation should include:

- Nonlinear cartpole dynamics.
- Direct multiple shooting or another clear nonlinear MPC transcription.
- IPOPT as the solver.
- State and control constraints (or we call it input constraints).
- A cost function that encourages the pole to reach the upright position while keeping the cart bounded.
- A simulation loop that repeatedly solves MPC, applies the first control action, and advances the cartpole state.
- At least one plot or gif animation (don't do screen recording) showing whether swing-up succeeded.

Adding Python packages to uv env, for example:

```bash
uv add casadi matplotlib pyyaml
```

But feel free to use any package in this task.

## Suggested MPC Design

You are expected to tune these values. The numbers below are starting points, not final answers.

- Time step: `0.02` to `0.05` seconds.
- Horizon length:
- Force limit: around `10` to `30` N, depending on your model parameters.
- Cart position limit: choose a realistic bound such as `[-2.4, 2.4]`.
- Use RK4 or another stable discretization method.
- Penalize distance from upright using wrapped angle error.
- Penalize cart position, cart velocity, pole angular velocity, and control effort.
- Use a stronger terminal cost than stage cost if needed.

Useful IPOPT options to consider:

```python
{
    "ipopt.print_level": 0,
    "ipopt.max_iter": 200,
    "ipopt.tol": 1e-4,
    "print_time": False,
}
```

## Written Question 1

### Why PID is not enough for cartpole swing-up in this setting?

Attach the answer to the PR.

## Written Question 2

### What might need to change to let PID also complete this task?

Attach the answer to the PR

## What To Submit In The PR

Your PR should include:

- Source code for the reusable MPC controller.
- Experiment code that can be run with `uv run ...`.
- A runnable entrypoint under `examples/`.
- A short experiment README explaining how to run your code.
- A brief results summary.
- Answers to the two written questions.

The most important thing is not perfect performance on the first try. The important thing is to show a clean workflow, a reasonable nonlinear MPC formulation, thoughtful tuning, and clear communication.
