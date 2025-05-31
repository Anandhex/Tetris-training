# Tetris ML-Agents Training

A reinforcement learning project that trains AI agents to play Tetris using Unity ML-Agents toolkit.

## Features

- Custom Tetris environment built in Unity
- Reinforcement learning agent training using ML-Agents
- Cross-platform Unity builds (Windows, macOS, Linux)
- TensorFlow and PyTorch support for neural networks
- Configurable training parameters
- Graphics mode for visualization and testing

## Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.9
- Unity 2021.3+ (if modifying the Tetris environment)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Anandhex/Tetris-training.git
cd Tetris-training
```

### 2. Create conda environment

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate ml-unity
```

### 3. Verify installation

```bash
# Check if environment is active
conda info --envs

# Test ML-Agents installation
python -c "import mlagents; print('ML-Agents installed successfully!')"

# Test other key dependencies
python -c "import torch, gym; print('PyTorch and Gym installed successfully!')"
```

## Key Dependencies

This environment includes:

- **ML-Agents 0.30.0** - Unity Machine Learning Agents Toolkit
- **TensorFlow/Keras** - Deep learning framework for neural networks
- **PyTorch 1.11.0** - Alternative deep learning framework
- **OpenAI Gym 0.26.2** - Reinforcement learning environment interface
- **PettingZoo 1.15.0** - Multi-agent reinforcement learning environments
- **NumPy** - Numerical computing library

## Usage

### Training Mode (Headless)

The `main.py` file handles all the setup and configuration automatically for training:

```bash
# Always activate the environment first
conda activate ml-unity

# Run the main script - it handles everything!
python main.py
```

### Graphics Mode (Visualization)

Use `with_graphics.py` to run the game with visual interface using preset configurations:

```bash
# Always activate the environment first
conda activate ml-unity

# Run with graphics for visualization and testing
python with_graphics.py
```

The graphics mode automatically:

- Loads configuration from `data_config.yaml`
- Runs the Unity build with graphics enabled
- Provides visual feedback for agent performance
- Ideal for testing trained models or demonstrations

### Advanced Usage

If the scripts support command-line arguments:

```bash
# Run training with default settings
python main.py

# Run graphics mode with default settings
python with_graphics.py
```

Both scripts automatically:

- Detect your operating system
- Select the appropriate Unity build (Windows/Mac/Linux)
- Set up ML-Agents configuration
- Handle execution with appropriate settings

## Project Structure

```
Tetris-training/
├── environment.yml          # Conda environment configuration
├── main.py                 # Main training script - handles headless training
├── with_graphics.py        # Graphics mode script - handles visual execution
├── config/                 # ML-Agents training configurations
│   └── various configs
├── data_config.yaml        # Configuration file for graphics mode
├── results/                # Training results and tensorboard logs
├── builds/                 # Unity builds for different platforms
│   ├── windows/
│   │   └── /tetris/tetris-multi.exe
│   ├── mac/
│   │   └── tetris.app
│   └── linux/
│       └── tetris
└── README.md              # This file
```

## Configuration Files

### Training Configuration

Modify `config/tetris_config.yaml` to adjust training parameters:

- Learning rate
- Batch size
- Network architecture
- Reward parameters
- Training duration

Example configuration sections:

```yaml
behaviors:
  TetrisAgent:
    trainer_type: ppo
    hyperparameters:
      learning_rate: 3.0e-4
      batch_size: 1024
    network_settings:
      hidden_units: 512
      num_layers: 3
```

### Graphics Mode Configuration

The `data_config.yaml` file contains preset configurations for graphics mode:

- Environment settings
- Model parameters
- Visualization options
- Performance settings

This configuration is automatically loaded by `with_graphics.py` to ensure consistent execution parameters.

## Monitoring Training

```bash
# View training progress with TensorBoard
tensorboard --logdir results/

# Check training summaries
mlagents-learn --help
```

## Development

### Adding New Features

```bash
# Install additional packages
conda activate ml-unity
pip install new_package

# Update environment file
conda env export --from-history > environment.yml
```

### Updating the Environment

```bash
# Update environment from modified environment.yml
conda env update -f environment.yml --prune
```

## Troubleshooting

### Common Issues

**Environment creation fails:**

- Update conda: `conda update conda`
- Try using mamba: `conda install mamba -c conda-forge` then `mamba env create -f environment.yml`

**Unity build not found:**

- Ensure the correct build path for your OS in the training script
- Check that the Unity build has executable permissions (Linux/macOS)

**Training doesn't start:**

- Run `python main.py` and check console output for specific error messages
- Verify ML-Agents installation: `python -c "import mlagents"`
- Ensure the Unity build runs independently
- Check that all file paths in main.py are correct

**Graphics mode issues:**

- Verify `data_config.yaml` exists and is properly formatted
- Ensure graphics drivers are up to date
- Check that the Unity build supports graphics mode
- Run `python with_graphics.py` and check console output for specific errors

**Low training performance:**

- Reduce batch size if running out of memory
- Adjust the number of parallel environments
- Monitor system resources during training

## Quick Commands

```bash
# Setup (one-time)
conda env create -f environment.yml
conda activate ml-unity

# Run training (headless)
conda activate ml-unity
python main.py

# Run with graphics (visualization)
conda activate ml-unity
python with_graphics.py

# Monitor training progress (if training mode)
tensorboard --logdir results/

# Update dependencies
conda env export --from-history > environment.yml
```

## Usage Scenarios

### For Training

Use `main.py` when you want to:

- Train new models from scratch
- Run headless training for performance
- Generate training data and logs
- Perform batch training sessions

### For Visualization

Use `with_graphics.py` when you want to:

- Test trained models visually
- Demonstrate agent performance
- Debug agent behavior
- Create presentations or videos

## Results

Training results, models, and TensorBoard logs are saved in the `results/` directory. Each training run creates a subdirectory with:

- Trained model files (`.nn` and `.onnx`)
- Training configuration
- Performance metrics
- TensorBoard event files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both training and graphics modes
5. Update relevant configuration files
6. Submit a pull request

## Cleanup

```bash
# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n ml-unity
```
