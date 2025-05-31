import os
import subprocess
import platform
import time
import logging
import json
from pathlib import Path
from ruamel.yaml import YAML
from dataclasses import dataclass
from typing import Optional

# GPU detection imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlagents_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def detect_gpu_config():
    """Detect available GPU configuration"""
    gpu_info = {
        'available': False,
        'device_count': 0,
        'devices': [],
        'memory_info': [],
        'cuda_available': False,
        'recommended_device': None
    }
    
    # Check CUDA availability
    if TORCH_AVAILABLE:
        gpu_info['cuda_available'] = torch.cuda.is_available()
        if gpu_info['cuda_available']:
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['available'] = True
            
            for i in range(gpu_info['device_count']):
                device_name = torch.cuda.get_device_name(i)
                gpu_info['devices'].append(device_name)
                
                # Get memory info
                memory_total = torch.cuda.get_device_properties(i).total_memory
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_free = memory_total - memory_reserved
                
                gpu_info['memory_info'].append({
                    'device': i,
                    'total_mb': memory_total // (1024 * 1024),
                    'free_mb': memory_free // (1024 * 1024),
                    'allocated_mb': memory_allocated // (1024 * 1024),
                    'reserved_mb': memory_reserved // (1024 * 1024)
                })
    
    # Enhanced GPU detection using nvidia-ml-py
    if NVML_AVAILABLE:
        try:
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                name = nvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Get memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpu_info['devices'][i] = {
                    'index': i,
                    'name': name,
                    'memory_total_mb': mem_info.total // (1024 * 1024),
                    'memory_free_mb': mem_info.free // (1024 * 1024),
                    'memory_used_mb': mem_info.used // (1024 * 1024),
                    'gpu_utilization': util.gpu,
                    'memory_utilization': util.memory
                }
                
        except Exception as e:
            logger.warning(f"NVML detection failed: {e}")
    
    # Recommend best GPU (least utilized with most free memory)
    if gpu_info['available'] and gpu_info['memory_info']:
        best_device = max(gpu_info['memory_info'], key=lambda x: x['free_mb'])
        gpu_info['recommended_device'] = best_device['device']
    
    return gpu_info

ENV_PATHS = {
    "Darwin": "./builds/mac/tetris.app",      # macOS
    "Windows": "./builds/windows/tetris/tetris-multi.exe",
    "Linux": "./builds/linux/tetris"
}

@dataclass
class TrainingConfig:
    """Configuration class for ML-Agents training"""
    CONFIG_PATH: str = "base_config.yaml"
    RESULTS_DIR: str = "results"
    ENV_PATH: str = ENV_PATHS.get(platform.system(), 
                        f"./builds/{platform.system().lower()}/tetris")
    RUN_ID: str = f"tetris_training_{int(time.time())}"
    MAX_STEPS: int = 500000
    
    # GPU Configuration
    USE_GPU: bool = True  # Enable GPU if available
    GPU_DEVICE: Optional[int] = None  # Specific GPU device (None = auto-select)
    GPU_MEMORY_FRACTION: float = 0.95  # Fraction of GPU memory to use
    FORCE_CPU: bool = False  # Force CPU usage even if GPU available
    
    # Training options
    NO_GRAPHICS: bool = True
    RESUME: bool = False
    FORCE: bool = False
    INITIALIZE_FROM: Optional[str] = None


class MLAgentsTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.gpu_info = detect_gpu_config()
        self.setup_gpu_config()
        self.setup_directories()
        
    def setup_gpu_config(self):
        """Setup GPU configuration based on detection"""
        logger.info("=== GPU Configuration ===")
        
        if self.config.FORCE_CPU:
            logger.info("GPU usage forced to CPU mode")
            self.gpu_config = {'use_gpu': False, 'device': 'cpu'}
            return
        
        if not self.gpu_info['available']:
            logger.info("No GPU detected, using CPU")
            self.gpu_config = {'use_gpu': False, 'device': 'cpu'}
            return
        
        if not self.config.USE_GPU:
            logger.info("GPU usage disabled in config")
            self.gpu_config = {'use_gpu': False, 'device': 'cpu'}
            return
        
        # Log GPU information
        logger.info(f"Found {self.gpu_info['device_count']} GPU(s):")
        for i, device in enumerate(self.gpu_info['devices']):
            if isinstance(device, dict):
                logger.info(f"  GPU {i}: {device['name']}")
                logger.info(f"    Memory: {device['memory_free_mb']:.0f}MB free / {device['memory_total_mb']:.0f}MB total")
                logger.info(f"    Utilization: GPU {device['gpu_utilization']}%, Memory {device['memory_utilization']}%")
            else:
                logger.info(f"  GPU {i}: {device}")
        
        # Select GPU device
        if self.config.GPU_DEVICE is not None:
            selected_device = self.config.GPU_DEVICE
            logger.info(f"Using specified GPU device: {selected_device}")
        elif self.gpu_info['recommended_device'] is not None:
            selected_device = self.gpu_info['recommended_device']
            logger.info(f"Auto-selected GPU device: {selected_device} (most free memory)")
        else:
            selected_device = 0
            logger.info("Using default GPU device: 0")
        
        self.gpu_config = {
            'use_gpu': True,
            'device': selected_device,
            'device_name': f"cuda:{selected_device}" if TORCH_AVAILABLE else f"gpu:{selected_device}"
        }
        
        logger.info(f"GPU configuration: {self.gpu_config}")
        
    def setup_directories(self):
        """Create necessary directories"""
        Path(self.config.RESULTS_DIR).mkdir(exist_ok=True)
    
    def update_config_for_gpu(self):
        """Update the training config file with GPU settings if needed"""
        try:
            config_path = Path(self.config.CONFIG_PATH)
            if not config_path.exists():
                logger.error(f"Config file not found: {config_path}")
                return False
            
            with open(config_path) as f:
                yaml_loader = YAML()
                config = yaml_loader.load(f)

            # Add GPU/Torch settings
            if "torch_settings" not in config:
                config["torch_settings"] = {}
            
            if self.gpu_config['use_gpu']:
                config["torch_settings"]["device"] = self.gpu_config['device_name']
                # Set memory fraction if using GPU
                if TORCH_AVAILABLE:
                    config["torch_settings"]["memory_fraction"] = self.config.GPU_MEMORY_FRACTION
                logger.info(f"Updated config to use GPU device {self.gpu_config['device']}")
            else:
                config["torch_settings"]["device"] = "cpu"
                logger.info("Updated config to use CPU")

            # Update max steps
            if "behaviors" in config:
                for behavior_name in config["behaviors"]:
                    config["behaviors"][behavior_name]["max_steps"] = self.config.MAX_STEPS

            # Write updated config to a temporary file
            temp_config_path = config_path.parent / f"temp_{config_path.name}"
            yaml_dumper = YAML()
            yaml_dumper.indent(mapping=2, sequence=4, offset=2)
            
            with open(temp_config_path, "w") as f:
                f.write('---\n')
                yaml_dumper.dump(config, f)

            return str(temp_config_path)
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return None

    def run_training(self):
        """Run ML-Agents training"""
        logger.info(f"Starting ML-Agents training with run ID: {self.config.RUN_ID}")
        logger.info(f"Using device: {'GPU ' + str(self.gpu_config['device']) if self.gpu_config['use_gpu'] else 'CPU'}")
        
        # Update config file with GPU settings
        config_file = self.update_config_for_gpu()
        if not config_file:
            config_file = self.config.CONFIG_PATH
            logger.warning("Using original config file without GPU updates")
        
        # Prepare environment variables for GPU
        env = os.environ.copy()
        
        if self.gpu_config['use_gpu']:
            # Set CUDA device if using GPU
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_config['device'])
            
            # Set additional GPU-related environment variables
            if TORCH_AVAILABLE:
                env["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0;8.6"  # Common architectures
            
            logger.info(f"Set CUDA_VISIBLE_DEVICES={self.gpu_config['device']}")
        else:
            # Force CPU usage
            env["CUDA_VISIBLE_DEVICES"] = ""
            logger.info("Forced CPU usage")

        # Build ML-Agents command
        cmd = [
            "mlagents-learn", config_file,
            "--run-id", self.config.RUN_ID,
            "--env", self.config.ENV_PATH
        ]
        
        # Add optional flags
        if self.config.NO_GRAPHICS:
            cmd.append("--no-graphics")
        
        if self.config.RESUME:
            cmd.append("--resume")
        
        if self.config.FORCE:
            cmd.append("--force")
        
        if self.config.INITIALIZE_FROM:
            cmd.extend(["--initialize-from", self.config.INITIALIZE_FROM])

        # Add GPU-specific ML-Agents flags if available
        if self.gpu_config['use_gpu']:
            cmd.extend(["--torch-device", self.gpu_config['device_name']])

        # Always add train flag
        cmd.append("--train")

        logger.info(f"Running command: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            
            # Run the training
            result = subprocess.run(
                cmd, 
                env=env,
                text=True
            )
            
            end_time = time.time()
            training_time = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"Training completed successfully in {training_time:.1f} seconds")
                self.log_training_summary()
            else:
                logger.error(f"Training failed with return code: {result.returncode}")
                return False
                
            return True

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
        finally:
            # Clean up temporary config file
            if config_file != self.config.CONFIG_PATH:
                try:
                    Path(config_file).unlink()
                except:
                    pass

    def log_training_summary(self):
        """Log a summary of the training session"""
        logger.info("\n" + "="*50)
        logger.info("TRAINING SUMMARY")
        logger.info("="*50)
        logger.info(f"Run ID: {self.config.RUN_ID}")
        logger.info(f"Environment: {self.config.ENV_PATH}")
        logger.info(f"Max Steps: {self.config.MAX_STEPS}")
        logger.info(f"GPU Used: {self.gpu_config['use_gpu']}")
        if self.gpu_config['use_gpu']:
            logger.info(f"GPU Device: {self.gpu_config['device']}")
        logger.info(f"Results saved to: {self.config.RESULTS_DIR}/{self.config.RUN_ID}")
        logger.info("="*50)


def main():
    """Main function to run training"""
    # You can modify these parameters as needed
    config = TrainingConfig(
        CONFIG_PATH="data_config.yaml",
        RUN_ID=f"tetris_training_{int(time.time())}",
        MAX_STEPS=500000,
        USE_GPU=True,
        NO_GRAPHICS=False
    )
    
    # Print configuration
    logger.info("Training Configuration:")
    logger.info(f"  Config File: {config.CONFIG_PATH}")
    logger.info(f"  Environment: {config.ENV_PATH}")
    logger.info(f"  Run ID: {config.RUN_ID}")
    logger.info(f"  Max Steps: {config.MAX_STEPS}")
    logger.info(f"  Use GPU: {config.USE_GPU}")
    logger.info(f"  No Graphics: {config.NO_GRAPHICS}")
    
    # Initialize and run training
    trainer = MLAgentsTrainer(config)
    
    try:
        success = trainer.run_training()
        if success:
            logger.info("Training completed successfully!")
        else:
            logger.error("Training failed!")
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()