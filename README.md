# Tetris ML-Agents Training

A reinforcement learning project that trains AI agents to play Tetris using Unity ML-Agents toolkit.

## Features

- Custom Tetris environment built in Unity
- Reinforcement learning agent training using ML-Agents
- Cross-platform Unity builds (Windows, macOS, Linux)
- TensorFlow and PyTorch support for neural networks
- Configurable training parameters

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

### Running the Project

The `main.py` file handles all the setup and configuration automatically:

```bash
# Always activate the environment first
conda activate ml-unity

# Run the main script - it handles everything!
python main.py
```

### Advanced Usage

If `main.py` supports command-line arguments:

```bash
# Run with default settings
python main.py

# Example of possible arguments (adjust based on your implementation)
python main.py --train          # Start training mode
python main.py --inference      # Run inference with trained model
python main.py --run-id my_run  # Custom run identifier
```

The main script automatically:

- Detects your operating system
- Selects the appropriate Unity build (Windows/Mac/Linux)
- Sets up ML-Agents configuration
- Handles training or inference modes

## Project Structure

```
Tetris-training/
├── environment.yml          # Conda environment configuration
├── main.py                 # Main script - handles all setup and execution
├── config/                 # ML-Agents training configurations
│   └── various configs
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

## Training Configuration

Modify `config/tetris_config.yaml` to adjust:

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

**Low training performance:**

- Reduce batch size if running out of memory
- Adjust the number of parallel environments
- Monitor system resources during training

## Quick Commands

```bash
# Setup (one-time)
conda env create -f environment.yml
conda activate ml-unity

# Run the project
conda activate ml-unity
python main.py

# Monitor training progress (if training mode)
tensorboard --logdir results/

# Update dependencies
conda env export --from-history > environment.yml
```

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
4. Test with the conda environment
5. Submit a pull request

## Cleanup

```bash
# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n ml-unity
```
