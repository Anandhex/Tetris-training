# Tetris DQN Training Framework

This repository provides a fully configurable pipeline for training Deep Q-Network (DQN) agents (standard, with noise, greedy) and Stable Baselines3 algorithms (PPO, DQN, A2C) to play a Unity-based Tetris environment. Features include:

- **Command-line driven** experiment setup
- **Cross-platform launcher** (Windows, macOS, Linux)
- **JSON-based configuration** for rewards, curriculum, hyperparameters, and multi-instance setups
- **Curriculum learning** support — easy levels up to full game
- **TensorBoard integration** for metrics and debugging

---

## Repository Structure

```plaintext
├── config.json                 # Master configuration (global settings, instances, unity paths)
├── train_tetris.py             # Entry-point: loads config & CLI flags, runs training
├── cross_platform_launcher.py  # Launcher: spins up Unity & trainer instances
├── improved_tetris_trainer.py  # Core Trainer class with curriculum & reward logic
├── tetris_client.py            # Unity socket client
├── dqn_agent.py                # Standard DQN implementation
├── dqn_agent_with_greedy.py    # Greedy and noise-enhanced DQN agents
├── sb3_agent.py                # Stable Baselines3 agent wrapper (PPO, DQN, A2C)
└── README.md                   # This file
```

---

## Requirements

- Python 3.8+
- Unity build of Tetris environment (see `unity_paths` in `config.json`)
- Python packages:

  ```bash
  pip install numpy matplotlib pandas torch tensorboard gymnasium stable-baselines3
  ```

---

## Configuration (`config.json`)

Defines global settings, per-agent hyperparameters, and one or more training instances. Example structure:

```jsonc
{
  /* ... your JSON config ... */
}
```

### Configuration Parameters Overview

| Parameter                                    | Section       | Description                                                                                          |
| -------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------- |
| `unity_paths`                                | Global        | Maps OS names to Unity executable paths for cross-platform launching.                                |
| `host`                                       | Global        | Host address for Unity–Python communication socket.                                                  |
| `log_dir`                                    | Global        | Directory where training and Unity logs are saved.                                                   |
| `reward_config.heuristic_stack_height_coeff` | Reward Config | Penalizes tall stacks by scaling stack height in heuristic reward component.                         |
| `reward_config.heuristic_line_coeff`         | Reward Config | Rewards line clears quadratically in heuristic component; higher value emphasizes multi-line clears. |
| `reward_config.heuristic_holes_coeff`        | Reward Config | Penalizes holes in the stack, encouraging denser, hole-free placements.                              |
| `reward_config.heuristic_bumpiness_coeff`    | Reward Config | Penalizes surface unevenness to promote flatter top surfaces.                                        |
| `reward_config.shaped_base`                  | Reward Config | Provides a small constant reward each step to encourage continued play.                              |
| `reward_config.shaped_holes_coeff`           | Reward Config | Penalty per hole in shaping component to further discourage holes.                                   |
| `reward_config.shaped_bumpiness_coeff`       | Reward Config | Penalty per unit of bumpiness in shaping component.                                                  |
| `reward_config.shaped_wells_coeff`           | Reward Config | Penalty for deep wells in the board to avoid tall column gaps.                                       |
| `reward_config.height_reward_threshold`      | Reward Config | Stack height below which additional bonus is applied.                                                |
| `reward_config.height_reward_multiplier`     | Reward Config | Bonus per row when stack height is under threshold, incentivizing low stacks.                        |
| `reward_config.height_penalty_threshold`     | Reward Config | Stack height above which a large penalty begins.                                                     |
| `reward_config.height_penalty_multiplier`    | Reward Config | Scales penalty quadratically for exceeding safe stack height.                                        |
| `reward_config.normalization_divisor`        | Reward Config | Divides raw reward to keep values in a stable, bounded range.                                        |
| `reward_config.clip_min`                     | Reward Config | Minimum reward after clipping to avoid large negative spikes.                                        |
| `reward_config.clip_max`                     | Reward Config | Maximum reward after clipping to cap positive rewards.                                               |
| `reward_config.lines_multiplier`             | Reward Config | Multiplier for lines cleared in shaping component to strongly incentivize line clears.               |
| `agent_params.learning_rate`                 | Agent Params  | Learning rate for native DQN/nuna updates; balances convergence speed vs. stability.                 |
| `agent_params.batch_size`                    | Agent Params  | Number of samples per training step, affecting gradient variance and compute efficiency.             |
| `agent_params.gamma`                         | Agent Params  | Discount factor for future rewards; closer to 1 values emphasize long-term planning.                 |
| `sb3_params.common.verbose`                  | SB3 Params    | Controls verbosity of SB3 training logs.                                                             |
| `sb3_params.common.device`                   | SB3 Params    | Selects CPU or GPU automatically for faster training.                                                |
| `sb3_params.common.seed`                     | SB3 Params    | Seed for reproducibility across runs.                                                                |
| `sb3_params.ppo.learning_rate`               | SB3 PPO       | Policy/value network step size; similar trade-offs as DQN learning_rate.                             |
| `sb3_params.ppo.n_steps`                     | SB3 PPO       | Number of environment interactions per rollout; affects bias-variance trade-off.                     |
| `sb3_params.ppo.batch_size`                  | SB3 PPO       | Mini-batch size for PPO updates; larger sizes improve stability but cost more memory.                |
| `sb3_params.ppo.n_epochs`                    | SB3 PPO       | Number of optimization passes over each rollout; increases sample efficiency.                        |
| `sb3_params.ppo.gamma`                       | SB3 PPO       | Discount factor for PPO; same role as in DQN.                                                        |
| `sb3_params.ppo.gae_lambda`                  | SB3 PPO       | GAE smoothing parameter; adjusts bias vs. variance in advantage estimation.                          |
| `sb3_params.ppo.clip_range`                  | SB3 PPO       | PPO clipping threshold to limit policy updates, enhancing stability.                                 |
| `sb3_params.ppo.ent_coef`                    | SB3 PPO       | Entropy coefficient to encourage exploration.                                                        |
| `sb3_params.ppo.vf_coef`                     | SB3 PPO       | Value function loss weight in PPO total loss.                                                        |
| `sb3_params.ppo.max_grad_norm`               | SB3 PPO       | Gradient clipping threshold for PPO to prevent exploding gradients.                                  |
| `sb3_params.dqn.learning_rate`               | SB3 DQN       | Learning rate for SB3 DQN Q-network.                                                                 |
| `sb3_params.dqn.buffer_size`                 | SB3 DQN       | Replay buffer capacity for experience replay stability.                                              |
| `sb3_params.dqn.learning_starts`             | SB3 DQN       | Number of steps to populate buffer before training begins.                                           |
| `sb3_params.dqn.batch_size`                  | SB3 DQN       | Samples per replay batch; trade-off between noise and compute.                                       |
| `sb3_params.dqn.train_freq`                  | SB3 DQN       | Frequency of training vs. acting.                                                                    |
| `sb3_params.dqn.gradient_steps`              | SB3 DQN       | Number of gradient updates per training call.                                                        |
| `sb3_params.dqn.target_update_interval`      | SB3 DQN       | Steps between target network updates for stability.                                                  |
| `sb3_params.dqn.exploration_fraction`        | SB3 DQN       | Portion of training over which epsilon decays.                                                       |
| `sb3_params.dqn.exploration_initial_eps`     | SB3 DQN       | Starting epsilon for SB3 DQN exploration.                                                            |
| `sb3_params.dqn.exploration_final_eps`       | SB3 DQN       | Final epsilon after decay completes.                                                                 |
| `sb3_params.a2c.learning_rate`               | SB3 A2C       | Actor-critic update step size.                                                                       |
| `sb3_params.a2c.n_steps`                     | SB3 A2C       | Time steps per rollout for A2C; smaller values mean more frequent updates.                           |
| `sb3_params.a2c.gamma`                       | SB3 A2C       | Discount factor for A2C.                                                                             |
| `sb3_params.a2c.gae_lambda`                  | SB3 A2C       | GAE lambda for advantage estimation.                                                                 |
| `sb3_params.a2c.ent_coef`                    | SB3 A2C       | Entropy coefficient to maintain exploration.                                                         |
| `sb3_params.a2c.vf_coef`                     | SB3 A2C       | Critic loss coefficient.                                                                             |
| `sb3_params.a2c.max_grad_norm`               | SB3 A2C       | Gradient clipping for A2C training stability.                                                        |
| `instances[].name`                           | Instances     | Unique identifier for each training run.                                                             |
| `instances[].port`                           | Instances     | Port for this Unity instance; must be unique to avoid conflicts.                                     |
| `instances[].agent`                          | Instances     | Agent type to train (`dqn`, `dqn_noise`, `greedy`, `nuna`, `sb3_ppo`, `sb3_dqn`, `sb3_a2c`).         |
| `instances[].episodes`                       | Instances     | Number of episodes to train for this run.                                                            |
| `instances[].curriculum`                     | Instances     | Toggle curriculum learning on/off for this specific run.                                             |
| `instances[].reward_config`                  | Instances     | Partial overrides of global reward_config for custom shaping per instance.                           |

````

> **Note:** Per-instance overrides can replace or extend any of `host`, `port`, `reward_config`, `agent_params`, or `sb3_params.common`.

---

## Why the Reward Configuration Differs

Over time, we refined the reward function to address several key challenges:

- **Sparse vs. Shaped Signals**:  A simple `lines_cleared^2` reward is too sparse and causes slow learning. By adding a constant `shaped_base` and per-move shaping components (holes, bumpiness, wells), the agent receives more frequent feedback to guide its policy earlier in training.

- **Balancing Long-Term vs. Short-Term**:  Blending a heuristic estimate (`heuristic_*` terms) with the actual score-based component over the first 10,000 steps (via the `alpha` blend) helps the agent bootstrap from domain knowledge before relying entirely on learned rewards.

- **Stable Magnitudes**:  Without normalization & clipping (`normalization_divisor`, `clip_min`, `clip_max`), raw rewards can explode or vanish, destabilizing training. These parameters keep the reward in a bounded range for reliable gradient updates.

- **Encouraging Risk Management**:  Height-based bonuses and penalties (`height_reward_*`, `height_penalty_*`) explicitly teach the agent to keep stacks low—an essential skill in Tetris that pure line-based rewards do not capture.

Together, these changes yield smoother, faster convergence and more human-like stacking behavior.

---

## Reward Function Structure

1. **Heuristic Component**
   \[h\] = *h_sh*⋅stack_height
   &nbsp;&nbsp;+ *h_l*⋅(lines_cleared²)
   &nbsp;&nbsp;+ *h_h*⋅holes
   &nbsp;&nbsp;+ *h_b*⋅bumpiness

2. **Score Component**
   score_comp = lines_cleared² (– death_penalty if game over)

3. **Shaped Component**
   shaped = base + lines⋅lines_multiplier – holes⋅sh_h – bumpiness⋅sh_b – wells⋅sh_w
   + bonus/penalty for stack height under/over thresholds

4. **Blend**
   α = min(1, step / 10000)
   raw = (1–α)⋅h + α⋅score_comp + shaped

5. **Normalize & Clip**
   reward = clip(raw / normalization_divisor, clip_min, clip_max)

_All intermediate values are logged to TensorBoard under `debug/*` and `reward/*` tags._

---

## Running Experiments

### Cross-Platform Launcher

```bash
python cross_platform_launcher.py config.json
````

- Reads `unity_paths` to start Unity builds
- Generates per-instance JSON by merging globals + overrides
- Launches Unity and Python trainers for each `instances` entry
- Logs to `log_dir/<instance>_unity.log` and `log_dir/<instance>_train.log`

### Single Run (Direct)

```bash
python train_tetris.py --config config.json --instance baseline_dqn
```

_(Assumes you extend `train_tetris.py` to accept an `--instance` flag that picks one of the named runs.)_

---

## Monitoring via TensorBoard

```bash
tensorboard --logdir runs
```

Metrics include per-episode score, reward breakdowns, feature counts, and curriculum progress.

---

## Contact

For issues or enhancements, please open an issue or reach out to the maintainer.
