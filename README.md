# Tetris DQN Training Framework

This repository provides a fully configurable pipeline for training Deep Q-Network (DQN) agents (standard, with noise, or greedy) to play a Unity-based Tetris environment. Features include:

- **Command-line driven** experiment setup
- **Cross-platform launcher** (Windows, macOS, Linux)
- **JSON-based configuration** for rewards, curriculum, and per-run settings
- **Curriculum learning** support — easy levels up to full game
- **TensorBoard integration** for metrics and debugging

---

## Repository Structure

```plaintext
├── config.json                 # Master configuration (global rewards, instances, unity paths)
├── train_tetris.py             # Entry-point: loads config & CLI flags, runs training
├── cross_platform_launcher.py  # Launcher: spins up Unity & trainer instances
├── improved_tetris_trainer.py  # Core Trainer class with curriculum & reward logic
├── tetris_client.py            # Unity socket client
├── dqn_agent.py                # Standard DQN implementation
├── dqn_agent_with_greedy.py    # Greedy agent implementation
└── README.md                   # This file
```

---

## Requirements

- Python 3.8+
- Unity build of Tetris environment (see `unity_paths` in `config.json`)
- Python packages:
  ```bash
  pip install numpy matplotlib pandas torch tensorboard
  ```

---

## Configuration (`config.json`)

Defines global settings and per-instance overrides.

```json
{
  "unity_paths": {
    "Windows": "builds/windows/tetris/tetris-multi.exe",
    "Linux": "builds/linux/tetris.x86_64",
    "Darwin": "builds/mac/tetris-full-curriculum-dummy.app/Contents/MacOS/tetris-multi"
  },
  "host": "127.0.0.1",
  "log_dir": "logs",
  "reward_config": {
    "heuristic_stack_height_coeff": -0.51,
    "heuristic_line_coeff": 5.0,
    "heuristic_holes_coeff": -0.36,
    "heuristic_bumpiness_coeff": -0.18,
    "shaped_base": 0.2,
    "shaped_holes_coeff": 0.25,
    "shaped_bumpiness_coeff": 0.1,
    "shaped_wells_coeff": 0.1,
    "height_reward_threshold": 10,
    "height_reward_multiplier": 0.5,
    "height_penalty_threshold": 16,
    "height_penalty_multiplier": 3.0,
    "normalization_divisor": 50.0,
    "clip_min": -1.0,
    "clip_max": 1.0,
    "lines_multiplier": 75
  },
  "instances": [
    {
      "name": "run1",
      "port": 12351,
      "agent": "dqn",
      "episodes": 2000,
      "curriculum": true
    }
  ]
}
```

- **`unity_paths`**: Map OS to Unity executable path.
- **`reward_config`**: Global reward parameters tailored to your desired heuristics and shaping. See details below.
- **`instances`**: Array of runs (name, port, agent type, episodes, curriculum toggle).

---

## Reward Configuration Details

Each key in `reward_config` influences how the agent is rewarded at each step:

| Key                            | Description                                                                                                       |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `heuristic_stack_height_coeff` | Multiplier for stack height penalty in the heuristic component; negative values penalize tall stacks.             |
| `heuristic_line_coeff`         | Coefficient for squared lines cleared in the heuristic component; higher rewards for clearing more lines at once. |
| `heuristic_holes_coeff`        | Multiplier for hole penalty in the heuristic component; negative values discourage creating holes.                |
| `heuristic_bumpiness_coeff`    | Multiplier for bumpiness penalty in the heuristic component; negative values favor smoother surfaces.             |
| `shaped_base`                  | Flat base reward added every step.                                                                                |
| `shaped_holes_coeff`           | Penalty per hole in shaping component; encourages minimizing holes.                                               |
| `shaped_bumpiness_coeff`       | Penalty per unit bumpiness in shaping component.                                                                  |
| `shaped_wells_coeff`           | Penalty per unit well depth in shaping component; discourages deep wells.                                         |
| `height_reward_threshold`      | Stack height below which additional shaped reward is applied.                                                     |
| `height_reward_multiplier`     | Reward per row under the threshold; encourages keeping stack low.                                                 |
| `height_penalty_threshold`     | Stack height above which penalty is applied.                                                                      |
| `height_penalty_multiplier`    | Penalty scaling above the threshold; punishes dangerously tall stacks.                                            |
| `normalization_divisor`        | Divisor to scale raw reward into a smaller range.                                                                 |
| `clip_min`                     | Minimum reward after clipping.                                                                                    |
| `clip_max`                     | Maximum reward after clipping.                                                                                    |
| `lines_multiplier`             | Additional multiplier for lines cleared in the shaped component; e.g., 75 for strong line incentives.             |

---

## Reward Function Structure

The reward is computed in five stages:

1. **Heuristic Component**  
   Uses board features and coefficients to estimate quality of a placement:
   ```text
   heuristic = (heuristic_stack_height_coeff * stack_height)
             + (heuristic_line_coeff * lines_cleared^2)
             + (heuristic_holes_coeff * holes)
             + (heuristic_bumpiness_coeff * bumpiness)
   ```
2. **Score Component**  
   Direct measure of lines cleared, with optional death penalty:
   ```text
   score_comp = lines_cleared^2
   if game_over: score_comp -= death_penalty
   ```
3. **Shaped Component**  
   Additional shaping to encourage low stacks and penalize holes/wells:
   ```text
   shaped = shaped_base
          + lines_cleared * lines_multiplier
          - holes * shaped_holes_coeff
          - bumpiness * shaped_bumpiness_coeff
   # extra row-based reward/penalty:
   if stack_height < height_reward_threshold:
       shaped += (height_reward_threshold - stack_height) * height_reward_multiplier
   if stack_height > height_penalty_threshold:
       shaped -= (stack_height - height_penalty_threshold)^2 * height_penalty_multiplier
   ```
4. **Blend Heuristic & Score**  
   Linearly blend heuristic and score components over time:
   ```text
   alpha = min(1, step / 10000)
   raw_reward = (1 - alpha)*heuristic + alpha*score_comp + shaped
   ```
5. **Normalization & Clipping**  
   Scale and clip to a bounded range:
   ```text
   normalized = raw_reward / normalization_divisor
   reward = clip(normalized, clip_min, clip_max)
   ```

All intermediate values (heuristic, score_comp, shaped, raw_reward, normalized, final reward) are logged to TensorBoard under corresponding tags for debugging and analysis.

--------------------------------|-------------|
| `heuristic_stack_height_coeff` | Multiplier for stack height penalty in the heuristic component; negative values penalize tall stacks. |
| `heuristic_line_coeff` | Coefficient for squared lines cleared in the heuristic component; higher rewards for clearing more lines at once. |
| `heuristic_holes_coeff` | Multiplier for hole penalty in the heuristic component; negative values discourage creating holes. |
| `heuristic_bumpiness_coeff` | Multiplier for bumpiness penalty in the heuristic component; negative values favor smoother surfaces. |
| `shaped_base` | Flat base reward added every step. |
| `shaped_holes_coeff` | Penalty per hole in shaping component; encourages minimizing holes. |
| `shaped_bumpiness_coeff` | Penalty per unit bumpiness in shaping component. |
| `shaped_wells_coeff` | Penalty per unit well depth in shaping component; discourages deep wells. |
| `height_reward_threshold` | Stack height below which additional shaped reward is applied. |
| `height_reward_multiplier` | Reward per row under the threshold; encourages keeping stack low. |
| `height_penalty_threshold` | Stack height above which penalty is applied. |
| `height_penalty_multiplier` | Penalty scaling above the threshold; punishes dangerously tall stacks. |
| `normalization_divisor` | Divisor to scale raw reward into a smaller range. |
| `clip_min` | Minimum reward after clipping. |
| `clip_max` | Maximum reward after clipping. |
| `lines_multiplier` | Additional multiplier for lines cleared in the shaped component; e.g., 75 for strong line incentives. |

---

## Example Instances

Here are some example `instances` you can run, demonstrating different agents, curriculum settings, and reward overrides:

```json
[
  {
    "name": "baseline_dqn",
    "port": 12351,
    "agent": "dqn",
    "episodes": 1000,
    "curriculum": true
  },
  {
    "name": "noisy_dqn",
    "port": 12352,
    "agent": "dqn_noise",
    "episodes": 1500,
    "curriculum": true,
    "reward_config": {
      "lines_multiplier": 100,
      "height_penalty_multiplier": 5.0
    }
  },
  {
    "name": "greedy_eval",
    "port": 12353,
    "agent": "greedy",
    "episodes": 500,
    "curriculum": false
  }
]
```

- **`baseline_dqn`**: Standard DQN with default rewards and full curriculum progression.
- **`noisy_dqn`**: DQN with exploration noise, stronger line reward and harsher height penalty to bias towards low stacks.
- **`greedy_eval`**: Greedy agent for evaluation; skips curriculum (starts at final stage) and runs fewer episodes.

## Training Workflow

### 1. Launcher (Cross-Platform)

Use `cross_platform_launcher.py` to spin up multiple experiments:

```bash
python cross_platform_launcher.py config.json
```

This will:

1. Read `config.json` and `unity_paths` for your OS.
2. Merge global + instance reward configs and write per-run `*_config.json`.
3. Launch each Unity player instance on its specified port.
4. Launch each Python trainer with `train_tetris.py --config <run_config>`.

Logs are saved under `logs/<instance>_unity.log` and `<instance>_train.log`.

### 2. Direct Training

Alternatively, run a single experiment via `train_tetris.py`:

```bash
python train_tetris.py \
  --config config.json \
  --host 127.0.0.1 \
  --port 12351 \
  --agent dqn \
  --episodes 2000 \
  --curriculum
```

---

## TensorBoard Integration

Metrics and debugging data are logged to the TensorBoard directory (e.g. `runs/tetris_dqn_XXXXXXXX`).

Launch TensorBoard:

```bash
tensorboard --logdir runs
```

---

## Contact

For issues or questions, please open an issue or contact the maintainer.
