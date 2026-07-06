# Task 007: Vehicle Residual Dynamics Learning for MPPI

## Purpose

Task 7 moves from the cartpole learned-dynamics exercise toward the vehicle setting.

The goal is to collect a diverse vehicle dynamics dataset, train a neural network residual model, and do an initial integration with MPPI. The actual simulator should still use the higher-fidelity dynamic bicycle model with tire forces. MPPI should use a simpler kinematic bicycle rollout model, then compensate its prediction error with a learned residual model.

The important idea is:

```text
actual simulator step function = dynamic bicycle with tire model
MPPI rollout dynamics = kinematic bicycle + learned residual correction
```

This task is not mainly about retuning MPPI. Reuse the best MPPI or Smooth MPPI controller parameters from Task 4 and Task 5 as much as possible. The main variable should be the learned residual dynamics model and the data used to train it.

## Learning Goals

By the end of this task, you should be able to:

- Collect diverse state/action transition data without manual driving.
- Mix expert, noisy expert, motion-primitive, and random behavior data.
- Train a residual dynamics model with a held-out validation split.
- Use state and action history as neural network input.
- Encode vehicle states in local coordinates instead of absolute world position.
- Handle angle inputs using sine and cosine features.
- Evaluate whether residual learning improves MPPI rollout predictions.
- Test learned residual MPPI under different friction conditions and ice patches.
- Explain where learned dynamics helps and where it remains unreliable.

## Reading and References

Use these references:

- Task 4 specification: `docs/tasks/004_fiala_trailer_ice_mppi.md`
- Task 5 specification: `docs/tasks/005_trailer_model_mismatch_mppi.md`
- Task 6 specification: `docs/tasks/006_online_learned_cartpole_mppi.md`
- MPPI, Smooth MPPI, and learned-dynamics papers in `docs/reading-list/papers.md`
- Learning Terrain-Aware Kinodynamic Model for Autonomous Off-Road Rally Driving With Model Predictive Path Integral Control: <https://ieeexplore.ieee.org/document/10258400>
- Bridging Active Exploration and Uncertainty-Aware Deployment Using Probabilistic Ensemble Neural Network Dynamics: <https://arxiv.org/abs/2305.12240>

Task 7 should build on the vehicle simulation and controller code from previous tasks. Do not reintroduce `uncertain-racecar-gym` as the simulator for this task.

## Repository Locations

Put reusable dataset and feature code here:

```text
src/learning/datasets/
```

Suggested files:

```text
src/learning/datasets/vehicle_dataset.py
src/learning/datasets/vehicle_history_windows.py
src/learning/datasets/vehicle_feature_transforms.py
```

Put reusable learned residual dynamics code here:

```text
src/learning/models/
```

Suggested files:

```text
src/learning/models/vehicle_residual_dynamics.py
src/learning/models/vehicle_residual_training.py
```

Put MPPI rollout adapters here:

```text
src/controllers/mpc/
```

Suggested files:

```text
src/controllers/mpc/vehicle_residual_rollout.py
src/controllers/mpc/vehicle_residual_mppi.py
```

Put experiment-specific code here:

```text
experiments/exp_007_vehicle_residual_dynamics/
```

Put runnable entrypoints here:

```text
examples/
```

Suggested runnable files:

```text
examples/run_vehicle_dataset_collection.py
examples/run_vehicle_residual_training.py
examples/run_vehicle_residual_mppi.py
examples/run_vehicle_residual_prediction_plot.py
```

## GitHub Workflow

1. Create a GitHub issue titled:

```text
Task 007: Vehicle Residual Dynamics Learning for MPPI
```

2. In the issue, briefly describe your plan:

- Which actual simulator dynamics you will use.
- Which kinematic rollout model you will use as the nominal model.
- How you will collect diverse data without manual driving.
- What friction levels you will include during data collection.
- What state/action history length `H` you will start with.
- What neural network architecture you will use.
- How you will split train and validation data.
- How the learned residual model will be inserted into MPPI rollouts.
- Which Task 4 or Task 5 MPPI parameters you will reuse.
- What plots and metrics you will report.

3. Create a branch from `main`. Use a branch name like:

```text
task-007-vehicle-residual-dynamics-mppi
```

4. Make small commits with clear messages. Example:

```text
Add vehicle dataset collection policies
Add vehicle history-window dataset
Add residual dynamics model training
Add learned residual MPPI rollout adapter
Document task 007 results
```

5. Open a pull request. Use this PR description template:

```markdown
## Summary

- Added diverse vehicle dataset collection.
- Added train/validation conversion with state-action history windows.
- Added a neural residual dynamics model.
- Integrated the residual model with kinematic MPPI rollouts.
- Evaluated residual MPPI under friction and ice-patch tests.

## Dataset Collection

- Actual simulator dynamics:
- Collection environments:
- Friction levels:
- Expert track-following policy:
- Open-space motion primitive policy:
- Noisy expert policy:
- Random policy:
- Number of trajectories per source:
- Total transitions:
- Dataset file locations:
- Why the dataset is diverse enough:

## State Representation

- Raw simulator state:
- Local-frame state used for learning:
- Action representation:
- Angle features encoded as `sin` and `cos`:
- Absolute world features excluded:
- History length `H`:
- Flattening order:
- Deployment-time history padding rule:

## Residual Model

- Nominal kinematic model:
- Residual target definition:
- Network architecture:
- Loss:
- Optimizer:
- Normalization:
- Batch size:
- Epochs:
- Best checkpoint selection rule:

## Train / Validation

- Train split size:
- Validation split size:
- Whether trajectories are split by trajectory or shuffled transition windows:
- Validation RMSE at the best checkpoint:
- Training RMSE at the best checkpoint:
- Saved checkpoint path:

## MPPI Integration

- Controller:
- Reused Task 4 or Task 5 parameter source:
- Horizon:
- Number of samples:
- Temperature:
- Noise covariance:
- Cost weights:
- Control limits:
- Rollout model without residual:
- Rollout model with residual:
- Actual simulator used for closed-loop evaluation:

## Results

### One-Step and Open-Loop Prediction

Include validation RMSE and open-loop prediction plots under multiple friction levels.

### Closed-Loop MPPI

Include a comparison table for kinematic MPPI and residual MPPI.

### Ice Patch Tests

Include tests with several ice patches. The actual simulator should use true local friction. Report whether the residual model improves, fails, or becomes unreliable.

## Media

- Closed-loop trajectory plot:
- Open-loop prediction plot:
- Success case link:
- Failure case link:

Do not commit large videos, GIFs, checkpoints, or datasets directly to git. Upload media to a GitHub issue or PR comment, then paste the uploaded media links here. Large datasets and checkpoints should be reproducible from scripts or stored outside git.

## Answers

### Why should the dataset mix expert, noisy expert, motion-primitive, and random behavior?

### Why should the neural network avoid absolute `x, y` world coordinates?

### Why do we encode angles with sine and cosine?

### What does the state/action history help the residual model infer?

### When does residual MPPI improve over pure kinematic MPPI, and when does it fail?

## Checklist

- [ ] Code runs with `uv run ...`
- [ ] Dataset is collected without manual driving
- [ ] Dataset mixes expert, noisy expert, motion-primitive, and random behavior
- [ ] Dataset includes multiple global friction levels
- [ ] Actual data-collection simulator uses dynamic bicycle dynamics with tire model
- [ ] Train/validation split is 70/30
- [ ] Validation RMSE is checked every epoch
- [ ] Best checkpoint is selected by validation performance
- [ ] Neural network input uses state/action history
- [ ] History length `H=4` is tested by default
- [ ] Absolute world `x, y` are excluded from neural network inputs
- [ ] Angle inputs are encoded with `sin(angle)` and `cos(angle)`
- [ ] Initial history is handled correctly during training and deployment
- [ ] Neural network predicts residual error relative to the kinematic model
- [ ] MPPI rollout uses kinematic bicycle plus learned residual
- [ ] Actual closed-loop simulator still uses dynamic bicycle dynamics with tire model
- [ ] MPPI hyperparameters mostly reuse Task 4 or Task 5 settings
- [ ] Ice-patch tests are included
- [ ] Open-loop prediction plots are included
- [ ] Closed-loop trajectory plots are included
- [ ] Large datasets, checkpoints, videos, and GIFs are not committed to git
- [ ] Written question answers are included
```

## Technical Requirements

Your implementation should include:

- Automated dataset collection in a large open environment.
- Optional dataset collection on the race track using an expert MPPI or Smooth MPPI policy.
- Multiple data sources: expert, noisy expert, open-space motion primitives, and random behavior.
- Multiple global friction levels during dataset collection.
- A train/validation split of `70/30`.
- Validation RMSE computed every epoch.
- Best checkpoint selected by validation performance, not training loss.
- A history-window dataset using state and action history.
- A local-frame neural network input representation.
- Sine/cosine encoding for all angle inputs.
- A residual target based on a kinematic bicycle prediction.
- MPPI integration using the kinematic bicycle model plus learned residual correction.
- Tests with ice patches after training.
- A short experiment README explaining how to reproduce the task.
- Answers to the written questions.

## Subtask 1: Dataset Collection

Collect a dataset in a large open environment. The goal is diversity, not a single clean racing behavior.

Use a mixture of these data sources:

- Expert track-following policy using MPPI or Smooth MPPI.
- Expert open-space motion primitives.
- Noisy expert rollouts with Gaussian noise added to the expert action.
- Pure random behavior.

Do not manually drive the vehicle to collect the dataset.

The open-space motion primitives should cover a wide range of behavior:

- Straight-line driving at different target speeds.
- Acceleration and deceleration profiles.
- Mild curves.
- Severe curves.
- Braking while turning.
- Low-speed and high-speed turning.
- Recovery-like behavior after noisy or unstable motion, if feasible.

The dataset should be broad enough that MPPI-style noisy rollouts are not completely out of distribution.

## Subtask 2: Friction Variation

Collect data under multiple global friction levels.

For data collection, you do not need circular ice patches. It is enough to set the entire open space or track to a fixed friction value, collect trajectories, then repeat for another friction value.

Suggested friction levels:

```text
mu = 1.0, 0.8, 0.6, 0.4
```

You may add more levels if it is easy. The reason for changing global friction during data collection is to let the state/action history indirectly encode friction-dependent behavior such as slip, yaw response, and delayed recovery.

## Subtask 3: History-Window Dataset

Use state and action history as the neural network input.

Start with:

```text
H = 4
```

The model input should be built from a sliding window:

```text
x_{t-H+1}, u_{t-H+1}, x_{t-H+2}, u_{t-H+2}, ..., x_t, u_t
```

Flatten the window in this exact time order. Use the same ordering during training and deployment.

For dataset conversion, you may drop the first `H-1` steps of each trajectory because those samples do not have enough history. During deployment, initialize missing history with zeros until enough real steps have been observed.

You may test additional history lengths, such as:

```text
H = 1, 2, 4, 8
```

However, `H=4` should be the default baseline.

## Subtask 4: Local State Representation

The neural network should not receive absolute world `x, y` coordinates.

Absolute position can cause the model to treat the same local dynamic state differently just because it happens at a different place on the map. Instead, use local-frame quantities such as:

- Longitudinal velocity.
- Lateral velocity.
- Yaw rate.
- Slip-related quantities, if available.
- Heading error or local-frame orientation features.
- Track-relative quantities, if used.
- Previous actions.

For every angle feature, do not feed a raw wrapped radian value directly if it crosses the `pi` to `-pi` discontinuity. Encode angle `theta` as:

```text
cos(theta), sin(theta)
```

Apply this rule consistently to heading, slip angle, hitch angle if used, and any other angular state.

## Subtask 5: Residual Target Definition

Use a kinematic bicycle model as the nominal rollout model.

For each transition in the dataset:

1. Take the current state `x_t` and action `u_t`.
2. Predict one step with the kinematic bicycle model:

```text
\bar{x}_{t+1} = f_{\text{kin}}(x_t, u_t)
```

3. Compare it with the true next state from the dynamic bicycle simulator:

```text
x_{t+1}
```

4. Train the neural network to predict the kinematic prediction error:

```text
e_t = \bar{x}_{t+1} - x_{t+1}
```

Then the learned rollout model is:

```text
x_{t+1}^{\text{pred}} = f_{\text{kin}}(x_t, u_t) - \hat{e}_\theta(\text{history}_{t})
```

If you use angle states in the prediction error, be careful with wrapping. Report exactly how angle errors are represented and reconstructed.

You may use a kinematic trailer model instead of the kinematic bicycle model if your implementation is ready for trailer residual learning. If you do this, clearly report the state, input, and residual definition.

## Subtask 6: Training and Checkpoint Selection

Split the dataset:

```text
70% train
30% validation
```

The validation set must not be used for training.

At each epoch, report:

- Training loss.
- Validation RMSE.
- Validation RMSE by state dimension, if useful.

Select the best checkpoint using validation performance, not training performance.

Save enough metadata to reproduce the checkpoint:

- Dataset version or collection command.
- Train/validation split seed.
- History length.
- Feature list.
- Normalization statistics.
- Network architecture.
- Optimizer settings.
- Best epoch.

Do not commit large checkpoint files directly to git. Store them in ignored output directories or describe how to regenerate them.

## Subtask 7: MPPI Integration

Integrate the learned residual model into MPPI.

Closed-loop evaluation should use:

```text
actual simulator: dynamic bicycle model with tire model
MPPI rollout: kinematic bicycle model + learned residual
```

Also compare against:

```text
MPPI rollout: kinematic bicycle model only
```

Use the same MPPI hyperparameters for both comparisons unless a very small change is absolutely necessary. Reuse the best Task 4 or Task 5 controller parameters. Do not run a new MPPI hyperparameter tuning sweep for this task.

Report:

- Which previous task parameters you reused.
- Whether any parameter changed.
- Why the change was necessary, if any.

## Subtask 8: Ice Patch Evaluation

After training on global friction datasets, test the controller with ice patches.

The actual simulator should use the true local friction when the vehicle enters an ice patch. The MPPI rollout should use the learned residual model and whatever friction information is available through the history; it should not be given future knowledge of an ice patch before the vehicle experiences it.

Compare at least:

- Kinematic MPPI without learned residual.
- Kinematic MPPI with learned residual.

Run multiple target speeds and at least a few trials per setting.

Report whether the residual model:

- Improves tracking before entering ice.
- Helps after friction changes.
- Becomes unreliable after abrupt friction changes.
- Recovers after enough history reflects the new friction condition.

## Subtask 9: Prediction and Trajectory Plots

Include plots that make the model behavior visible.

At minimum, include:

- Actual closed-loop trajectory.
- Open-loop prediction from kinematic model only.
- Open-loop prediction from kinematic model plus neural residual.
- Prediction comparison under multiple friction levels.
- Prediction comparison near or after an ice patch.

Useful plots:

- Position trajectory in local or world frame.
- Speed over time.
- Yaw rate over time.
- Slip angle over time.
- One-step residual prediction error.
- Open-loop error versus rollout horizon.
- Validation RMSE over epochs.

## Result Tables

Create tables that make the comparison easy to read.

At minimum, include:

- Controller.
- Rollout model.
- Actual simulator model.
- Dataset version.
- History length.
- Friction condition.
- Target speed.
- Number of trials.
- Success rate.
- Average speed.
- Average lateral error.
- Average slip angle.
- Average yaw rate.
- Open-loop prediction RMSE.
- Notes on failure mode.

Recommended rollout model labels:

```text
Kinematic bicycle
Kinematic bicycle + learned residual
```

If you also test trailer residual learning, add labels such as:

```text
Kinematic trailer
Kinematic trailer + learned residual
```

## Written Question 1

### Why should the dataset mix expert, noisy expert, motion-primitive, and random behavior?

Attach the answer to the PR.

Discuss coverage of the state/action space, MPPI's noisy sampled rollouts, distribution shift, and why only clean expert track-following data is not enough.

## Written Question 2

### Why should the neural network avoid absolute `x, y` world coordinates?

Attach the answer to the PR.

Discuss translation invariance, local dynamics, generalization to new parts of the track, and why the same dynamic state should not look different only because of global position.

## Written Question 3

### Why do we encode angles with sine and cosine?

Attach the answer to the PR.

Discuss the discontinuity between `pi` and `-pi`, smooth learning targets, and how the network should see nearby angles as nearby features.

## Written Question 4

### What does the state/action history help the residual model infer?

Attach the answer to the PR.

Discuss unobserved or indirect quantities such as friction, slip buildup, delayed tire response, actuator effects, and recent motion trends.

## Written Question 5

### When does residual MPPI improve over pure kinematic MPPI, and when does it fail?

Attach the answer to the PR.

Discuss one-step accuracy, open-loop compounding error, abrupt friction changes, out-of-distribution trajectories, and why receding-horizon control can help but cannot fully remove model error.

## What To Submit In The PR

Your PR should include:

- Source code for automated dataset collection.
- Source code for the history-window dataset.
- Source code for local-frame feature conversion.
- Source code for the neural residual model.
- Source code for training with a 70/30 train/validation split.
- Source code for validation RMSE logging and best-checkpoint selection.
- Source code for learned-residual MPPI rollout integration.
- Experiment code that can be run with `uv run ...`.
- Runnable entrypoints under `examples/`.
- A short experiment README explaining how to run everything.
- Result tables for prediction and closed-loop control.
- Plots showing open-loop prediction and closed-loop trajectories.
- Links to representative media, if useful.
- Answers to the written questions.

Large datasets, checkpoints, videos, and GIFs should not be committed directly to git. Use ignored output folders, external storage, or GitHub-uploaded media links.
