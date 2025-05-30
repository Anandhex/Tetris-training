import os
import subprocess
import platform
import random
import time
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from deap import base, creator, tools, algorithms
from tensorboard.backend.event_processing import event_accumulator
from ruamel.yaml import YAML
import pickle
from pathlib import Path
import torch

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
        logging.FileHandler('ga_optimization.log'),
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
                
                gpu_info['devices'].append({
                    'index': i,
                    'name': name,
                    'memory_total_mb': mem_info.total // (1024 * 1024),
                    'memory_free_mb': mem_info.free // (1024 * 1024),
                    'memory_used_mb': mem_info.used // (1024 * 1024),
                    'gpu_utilization': util.gpu,
                    'memory_utilization': util.memory
                })
                
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
class GAConfig:
    """Configuration class for GA parameters with GPU support"""
    CONFIG_TEMPLATE_PATH: str = "base_config.yaml"
    CONFIG_DIR: str = "configs"
    RESULTS_DIR: str = "results"
    ENV_PATH: str = ENV_PATHS.get(platform.system(), 
                        f"./builds/{platform.system().lower()}/tetris")
    MAX_STEPS: int = 500000
    POP_SIZE: int = 20
    N_GEN: int = 15
    CHECKPOINT_FREQ: int = 3
    PARALLEL_WORKERS: int = min(4, mp.cpu_count())  # Limit parallel jobs
    EARLY_STOPPING_PATIENCE: int = 5
    EARLY_STOPPING_THRESHOLD: float = 0.01
    
    # GPU Configuration
    USE_GPU: bool = True  # Enable GPU if available
    GPU_DEVICE: Optional[int] = None  # Specific GPU device (None = auto-select)
    GPU_MEMORY_FRACTION: float = 0.95  # Fraction of GPU memory to use
    FORCE_CPU: bool = False  # Force CPU usage even if GPU available
    
    # Adaptive parameters
    INITIAL_MUTATION_RATE: float = 0.3
    FINAL_MUTATION_RATE: float = 0.1
    CROSSOVER_RATE: float = 0.7
    TOURNAMENT_SIZE: int = 3
    ELITE_SIZE: int = 2  # Number of best individuals to preserve

# Enhanced parameter bounds with better defaults
PARAM_BOUNDS = {
    # PPO Hyperparameters - refined ranges based on best practices
    "learning_rate": (3e-5, 3e-4),  # Narrower range around typical values
    "batch_size": (256, 1024),      # Larger batches for stability
    "beta": (5e-4, 0.01),          # Entropy coefficient
    "epsilon": (0.15, 0.25),       # PPO clip parameter - narrower range
    "lambd": (0.92, 0.98),         # GAE lambda - typical range
    "num_epoch": (3, 10),          # Reduced upper bound
    "buffer_size": (10000, 50000), # Added buffer size parameter
    
    # Reward Weights - grouped by importance
    # Core gameplay rewards
    "clearReward": (0.8, 1.5),
    "comboMultiplier": (1.0, 2.5),
    "tetrisClearRewardMultiplier": (3.0, 5.0),
    "tripleLineClearRewardMultiplier": (2.0, 4.0),
    "doubleLineClearRewardMultiplier": (1.0, 2.5),
    
    # Special moves
    "perfectClearBonus": (0.0, 3.0),
    "tSpinReward": (0.0, 2.0),
    "iPieceGapFillBonus": (0.0, 1.5),
    
    # Penalties (more conservative ranges)
    "deathPenalty": (-3.0, -0.5),
    "stagnationPenaltyFactor": (0.0, 0.5),
    "stackHeightPenalty": (0.0, 0.5),
    "holeCreationPenalty": (0.0, 2.0),
    "uselessRotationPenalty": (0.0, 0.5),
    "idleActionPenalty": (-0.5, 0.0),
    
    # Board state rewards
    "roughnessRewardMultiplier": (0.0, 1.0),
    "roughnessPenaltyMultiplier": (0.0, 1.0),
    "holeFillReward": (0.0, 2.0),
    "wellRewardMultiplier": (0.0, 1.5),
    "iPieceInWellBonus": (0.0, 1.5),
    "maxWellRewardCap": (2.0, 8.0),
    
    # Accessibility and movement
    "accessibilityRewardMultiplier": (0.0, 1.0),
    "accessibilityPenaltyMultiplier": (0.0, 1.0),
    "moveDownActionReward": (0.0, 0.5),
    "hardDropActionReward": (0.0, 1.0),
}

# DEAP setup - handle creator classes properly
def setup_creator_classes():
    """Setup creator classes, handling existing ones properly"""
    # Remove existing classes if they exist
    for class_name in ["FitnessMax", "Individual"]:
        if hasattr(creator, class_name):
            delattr(creator, class_name)
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", dict, fitness=creator.FitnessMax, generation=int)

# Call this at module level to ensure classes are available
setup_creator_classes()

# Integer parameters for proper handling
INTEGER_PARAMS = {"batch_size", "num_epoch", "buffer_size"}

class OptimizedGA:
    def __init__(self, config: GAConfig):
        self.config = config
        self.gpu_info = detect_gpu_config()
        self.setup_gpu_config()
        self.setup_deap()
        self.setup_directories()
        self.best_fitness_history = []
        self.stagnation_counter = 0
        
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
        for dir_path in [self.config.CONFIG_DIR, self.config.RESULTS_DIR, "ga_results", "checkpoints"]:
            Path(dir_path).mkdir(exist_ok=True)
    
    def setup_deap(self):
        """Setup DEAP toolbox with improved operators"""
        # Creator classes are already set up at module level
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.generate_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.smart_crossover)
        self.toolbox.register("mutate", self.adaptive_mutation)
        self.toolbox.register("select", tools.selTournament, tournsize=self.config.TOURNAMENT_SIZE)
        self.toolbox.register("evaluate", self.evaluate_individual)

    def generate_individual(self) -> creator.Individual:
        """Generate individual with improved parameter sampling"""
        individual = {}
        for key, bounds in PARAM_BOUNDS.items():
            if key in INTEGER_PARAMS:
                individual[key] = random.randint(bounds[0], bounds[1])
            else:
                # Use beta distribution for better parameter exploration
                if random.random() < 0.8:  # 80% of time use beta distribution
                    alpha, beta = 2, 2  # Concentrates values toward middle
                    normalized = np.random.beta(alpha, beta)
                    individual[key] = bounds[0] + normalized * (bounds[1] - bounds[0])
                else:  # 20% uniform sampling for exploration
                    individual[key] = random.uniform(bounds[0], bounds[1])
        
        ind = creator.Individual(individual)
        ind.generation = 0
        return ind

    def smart_crossover(self, ind1: creator.Individual, ind2: creator.Individual) -> Tuple:
        """Improved crossover with parameter grouping"""
        # Group related parameters for block crossover
        param_groups = [
            ["learning_rate", "batch_size", "beta", "epsilon", "lambd", "num_epoch"],
            ["clearReward", "comboMultiplier", "tetrisClearRewardMultiplier"],
            ["deathPenalty", "stagnationPenaltyFactor", "stackHeightPenalty"],
            # Individual parameters get uniform crossover
        ]
        
        for group in param_groups:
            if random.random() < 0.3:  # 30% chance for block crossover
                for param in group:
                    if param in ind1:
                        ind1[param], ind2[param] = ind2[param], ind1[param]
        
        # Uniform crossover for remaining parameters
        ungrouped_params = set(ind1.keys()) - {p for group in param_groups for p in group}
        for param in ungrouped_params:
            if random.random() < 0.5:
                ind1[param], ind2[param] = ind2[param], ind1[param]
        
        return ind1, ind2

    def adaptive_mutation(self, individual: creator.Individual, generation: int = 0) -> Tuple:
        """Adaptive mutation that decreases over generations"""
        # Calculate adaptive mutation rate
        progress = generation / self.config.N_GEN if self.config.N_GEN > 0 else 0
        current_mutation_rate = (
            self.config.INITIAL_MUTATION_RATE * (1 - progress) + 
            self.config.FINAL_MUTATION_RATE * progress
        )
        
        for key, bounds in PARAM_BOUNDS.items():
            if random.random() < current_mutation_rate:
                if key in INTEGER_PARAMS:
                    # Smaller mutations for integers as generations progress
                    max_delta = max(1, int((bounds[1] - bounds[0]) * 0.1 * (1 - progress)))
                    delta = random.randint(-max_delta, max_delta)
                    individual[key] = np.clip(
                        individual[key] + delta, bounds[0], bounds[1]
                    )
                else:
                    # Adaptive gaussian noise
                    current_range = bounds[1] - bounds[0]
                    sigma = current_range * 0.1 * (1 - progress * 0.5)  # Decrease noise over time
                    new_val = individual[key] + random.gauss(0, sigma)
                    individual[key] = np.clip(new_val, bounds[0], bounds[1])
        
        return individual,

    def write_config(self, ind: creator.Individual, run_id: str) -> str:
        """Write configuration with GPU support and better error handling"""
        try:
            with open(self.config.CONFIG_TEMPLATE_PATH) as f:
                yaml_loader = YAML()
                config = yaml_loader.load(f)

            # Update PPO hyperparameters
            hparams = config["behaviors"]["TetrisAgent"]["hyperparameters"]
            hparams["learning_rate"] = float(ind["learning_rate"])
            hparams["batch_size"] = int(ind["batch_size"])
            hparams["beta"] = float(ind["beta"])
            hparams["epsilon"] = float(ind.get("epsilon", 0.2))
            hparams["lambd"] = float(ind.get("lambd", 0.95))
            hparams["num_epoch"] = int(ind.get("num_epoch", 3))
            
            if "buffer_size" in ind:
                hparams["buffer_size"] = int(ind["buffer_size"])
            
            config["behaviors"]["TetrisAgent"]["max_steps"] = self.config.MAX_STEPS

            # Add GPU/Torch settings
            if "torch_settings" not in config:
                config["torch_settings"] = {}
            
            if self.gpu_config['use_gpu']:
                config["torch_settings"]["device"] = self.gpu_config['device_name']
                # Set memory fraction if using GPU
                if TORCH_AVAILABLE:
                    config["torch_settings"]["memory_fraction"] = self.config.GPU_MEMORY_FRACTION
                logger.info(f"Config {run_id}: Using GPU device {self.gpu_config['device']}")
            else:
                config["torch_settings"]["device"] = "cpu"
                logger.info(f"Config {run_id}: Using CPU")

            # Add environment parameters (reward weights)
            if "environment_parameters" not in config:
                config["environment_parameters"] = {}

            reward_params = [k for k in PARAM_BOUNDS.keys() if k not in 
                           {"learning_rate", "batch_size", "beta", "epsilon", "lambd", "num_epoch", "buffer_size"}]

            for param in reward_params:
                if param in ind:
                    config["environment_parameters"][param] = float(ind[param])

            # Write config
            config_path = Path(self.config.CONFIG_DIR) / f"run_{run_id}.yaml"
            yaml_dumper = YAML()
            yaml_dumper.indent(mapping=2, sequence=4, offset=2)
            
            with open(config_path, "w") as f:
                f.write('---\n')
                yaml_dumper.dump(config, f)

            return str(config_path)
            
        except Exception as e:
            logger.error(f"Failed to write config for {run_id}: {e}")
            raise

    def train_and_evaluate(self, ind: creator.Individual, run_id: str) -> float:
        try:
            config_path = self.write_config(ind, run_id)
            logger.info(f"Starting training for Run {run_id}")

            # Prepare environment variables for GPU
            env = os.environ.copy()
            
            if self.gpu_config['use_gpu']:
                # Set CUDA device if using GPU
                env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_config['device'])
                
                # Set additional GPU-related environment variables
                if TORCH_AVAILABLE:
                    env["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0;8.6"  # Common architectures
                
                logger.info(f"Set CUDA_VISIBLE_DEVICES={self.gpu_config['device']} for {run_id}")
            else:
                # Force CPU usage
                env["CUDA_VISIBLE_DEVICES"] = ""
                logger.info(f"Forced CPU usage for {run_id}")

            cmd = [
                "mlagents-learn", config_path,
                "--run-id", run_id,
                "--env", self.config.ENV_PATH,
                "--no-graphics",
                "--train"
            ]

            # Add GPU-specific ML-Agents flags if available
            if self.gpu_config['use_gpu']:
                # ML-Agents will automatically use GPU if available and configured
                cmd.extend(["--torch-device", self.gpu_config['device_name']])

            # Add timeout to prevent hanging
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                timeout=3600,  # 1 hour timeout
                env=env
            )

            if result.returncode != 0:
                logger.error(f"Training failed for {run_id}")
                logger.error(f"STDERR: {result.stderr}")
                logger.error(f"STDOUT: {result.stdout}")
                return 0.0

            # Wait a moment for files to be written
            time.sleep(2)
            
            fitness = self.read_fitness_from_logs(run_id)
            logger.info(f"Training completed for {run_id}, fitness: {fitness}")
            
            return fitness

        except subprocess.TimeoutExpired:
            logger.error(f"Training timeout for {run_id}")
            return 0.0
        except Exception as e:
            logger.error(f"Training error for {run_id}: {e}")
            return 0.0

    def evaluate_individual(self, individual: creator.Individual) -> Tuple[float]:
        """Evaluate individual with generation tracking"""
        run_id = f"GA_Gen{getattr(individual, 'generation', 0)}_T{int(time.time())}_{random.randint(1000, 9999)}"
        fitness = self.train_and_evaluate(individual, run_id)
        return (fitness,)

    def save_checkpoint(self, population: List, generation: int, hof):
        """Save checkpoint for resuming"""
        checkpoint_data = {
            'generation': generation,
            'population': population,
            'hall_of_fame': list(hof),
            'best_fitness_history': self.best_fitness_history,
            'config': self.config.__dict__,
            'gpu_config': self.gpu_config,
            'gpu_info': self.gpu_info
        }
        
        checkpoint_path = Path("checkpoints") / f"checkpoint_gen_{generation}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_results(self, hof, generation: int):
        """Save results with enhanced formatting"""
        results_data = {
            'generation': generation,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_config': self.gpu_config,
            'best_individuals': [],
            'fitness_history': self.best_fitness_history,
            'config': self.config.__dict__
        }
        
        for i, ind in enumerate(hof):
            results_data['best_individuals'].append({
                "rank": i + 1,
                "fitness": float(ind.fitness.values[0]),
                "generation_found": getattr(ind, 'generation', 0),
                "parameters": {k: float(v) if isinstance(v, (int, float)) else v 
                             for k, v in dict(ind).items()}
            })
        
        results_path = Path("ga_results") / f"results_gen_{generation}.json"
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved: {results_path}")

    def check_early_stopping(self, current_best: float) -> bool:
        """Check if early stopping criteria are met"""
        if len(self.best_fitness_history) < 2:
            return False
        
        recent_best = max(self.best_fitness_history[-self.config.EARLY_STOPPING_PATIENCE:])
        if current_best - recent_best < self.config.EARLY_STOPPING_THRESHOLD:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        return self.stagnation_counter >= self.config.EARLY_STOPPING_PATIENCE

    def run_evolution(self):
        """Main evolution loop with GPU support"""
        logger.info(f"Starting GA with {self.config.POP_SIZE} individuals for {self.config.N_GEN} generations")
        logger.info(f"Using device: {'GPU ' + str(self.gpu_config['device']) if self.gpu_config['use_gpu'] else 'CPU'}")
        
        # Initialize population
        pop = self.toolbox.population(n=self.config.POP_SIZE)
        hof = tools.HallOfFame(5)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", lambda x: np.mean([v[0] for v in x]))
        stats.register("std", lambda x: np.std([v[0] for v in x]))
        stats.register("min", lambda x: np.min([v[0] for v in x]))
        stats.register("max", lambda x: np.max([v[0] for v in x]))
        
        start_time = time.time()
        
        for gen in range(self.config.N_GEN):
            logger.info(f"Generation {gen + 1}/{self.config.N_GEN}")
            gen_start_time = time.time()
            
            # Set generation for individuals
            for ind in pop:
                ind.generation = gen
            
            # Sequential evaluation (ML-Agents doesn't play well with parallel GPU usage)
            fitnesses = map(self.toolbox.evaluate, pop)
            
            # Assign fitness values
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            
            # Update hall of fame and statistics
            hof.update(pop)
            record = stats.compile(pop)
            
            gen_time = time.time() - gen_start_time
            logger.info(f"Gen {gen + 1} completed in {gen_time:.1f}s - "
                       f"Max: {record['max']:.2f}, Avg: {record['avg']:.2f}, "
                       f"Std: {record['std']:.2f}, Min: {record['min']:.2f}")
            
            self.best_fitness_history.append(record['max'])
            
            # Early stopping check
            if self.check_early_stopping(record['max']):
                logger.info(f"Early stopping triggered at generation {gen + 1}")
                break
            
            # Save checkpoint
            if (gen + 1) % self.config.CHECKPOINT_FREQ == 0:
                self.save_checkpoint(pop, gen + 1, hof)
                self.save_results(hof, gen + 1)
            
            # Evolution for next generation
            if gen < self.config.N_GEN - 1:
                # Elitism: preserve best individuals
                next_pop = tools.selBest(pop, self.config.ELITE_SIZE)
                
                # Fill rest with offspring
                offspring_size = self.config.POP_SIZE - self.config.ELITE_SIZE
                offspring = self.toolbox.select(pop, offspring_size)
                offspring = [self.toolbox.clone(ind) for ind in offspring]
                
                # Apply crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.config.CROSSOVER_RATE:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
                
                # Apply mutation
                for mutant in offspring:
                    if random.random() < self.config.INITIAL_MUTATION_RATE:
                        self.toolbox.mutate(mutant, generation=gen)
                        del mutant.fitness.values
                
                pop[:] = next_pop + offspring
        
        total_time = time.time() - start_time
        logger.info(f"Evolution completed in {total_time:.1f} seconds")
        
        # Final results
        self.save_results(hof, gen + 1)
        self.print_final_results(hof)
        
        return hof, pop

    def print_final_results(self, hof):
        """Print final results summary"""
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"GPU Configuration: {self.gpu_config}")
        
        for i, ind in enumerate(hof):
            logger.info(f"\nRank {i+1}: Fitness = {ind.fitness.values[0]:.3f}")
            logger.info(f"Generation found: {getattr(ind, 'generation', 'Unknown')}")
            
            # Print key parameters
            key_params = ["learning_rate", "batch_size", "clearReward", "tetrisClearRewardMultiplier", "deathPenalty"]
            params_str = ", ".join([f"{k}: {ind.get(k, 'N/A'):.4f}" if isinstance(ind.get(k), float) 
                                  else f"{k}: {ind.get(k, 'N/A')}" for k in key_params if k in ind])
            logger.info(f"Key parameters: {params_str}")

    def read_fitness_from_logs(self, run_id: str) -> float:
    
        possible_log_dirs = [
            Path(self.config.RESULTS_DIR) / run_id / "TetrisAgent",
            Path(self.config.RESULTS_DIR) / run_id,
            Path("results") / run_id / "TetrisAgent",  # Default ML-Agents path
            Path("results") / run_id,
            Path(".") / "results" / run_id / "TetrisAgent",
            Path(".") / "results" / run_id,
        ]
        
        log_dir = None
        event_files = []
        
        # Debug: Print what we're looking for
        logger.info(f"Searching for logs for run_id: {run_id}")
        
        for dir_path in possible_log_dirs:
            logger.debug(f"Checking directory: {dir_path}")
            if dir_path.exists():
                logger.debug(f"Directory exists: {dir_path}")
                # Look for TensorBoard event files
                pattern_files = list(dir_path.glob("events.out.tfevents.*"))
                if pattern_files:
                    log_dir = str(dir_path)
                    event_files = pattern_files
                    logger.info(f"Found event files in: {log_dir}")
                    logger.debug(f"Event files: {[f.name for f in event_files]}")
                    break
                else:
                    logger.debug(f"No event files in: {dir_path}")
            else:
                logger.debug(f"Directory does not exist: {dir_path}")
        
        if not log_dir:
            # Try a more aggressive search
            logger.warning(f"Standard paths failed, searching more broadly...")
            
            # Search in current directory and subdirectories
            for root_path in [Path("."), Path("results"), Path(self.config.RESULTS_DIR)]:
                if root_path.exists():
                    # Use recursive glob to find any event files containing the run_id
                    recursive_files = list(root_path.rglob("events.out.tfevents.*"))
                    for event_file in recursive_files:
                        if run_id in str(event_file.parent):
                            log_dir = str(event_file.parent)
                            event_files = [event_file]
                            logger.info(f"Found event files via recursive search in: {log_dir}")
                            break
                    if log_dir:
                        break
        
        if not log_dir:
            logger.error(f"No log directory found for {run_id} after exhaustive search")
            
            # Debug: List what directories actually exist
            results_base = Path(self.config.RESULTS_DIR)
            if results_base.exists():
                logger.info(f"Contents of {results_base}:")
                for item in results_base.iterdir():
                    logger.info(f"  {item.name} ({'dir' if item.is_dir() else 'file'})")
                    if item.is_dir() and run_id in item.name:
                        logger.info(f"    Contents of {item}:")
                        for subitem in item.iterdir():
                            logger.info(f"      {subitem.name} ({'dir' if subitem.is_dir() else 'file'})")
            
            return 0.0

        try:
            # Try to load the EventAccumulator
            ea = event_accumulator.EventAccumulator(log_dir)
            ea.Reload()
            
            scalar_tags = ea.Tags().get('scalars', [])
            logger.debug(f"Available scalar tags: {scalar_tags}")
            
            if not scalar_tags:
                logger.warning(f"No scalar tags found for {run_id}")
                return 0.0

            # Priority order for reward tags (expanded list)
            reward_tags = [
                "Environment/Cumulative Reward",
                "TetrisAgent/Environment/Cumulative Reward", 
                "Policy/Extrinsic Reward",
                "Policy/Extrinsic Value Estimate",
                "Environment/Episode Length",
                "Cumulative Reward",
                "Extrinsic Reward",
                "Episode Reward"
            ]
            
            reward_tag = None
            for tag in reward_tags:
                if tag in scalar_tags:
                    reward_tag = tag
                    logger.info(f"Using reward tag: {reward_tag}")
                    break
            
            # Fallback: find any reward-related tag
            if not reward_tag:
                for tag in scalar_tags:
                    if any(keyword in tag.lower() for keyword in ["reward", "cumulative", "extrinsic"]):
                        reward_tag = tag
                        logger.info(f"Using fallback reward tag: {reward_tag}")
                        break

            if not reward_tag:
                logger.warning(f"No suitable reward tag found for {run_id}")
                logger.warning(f"Available tags were: {scalar_tags}")
                return 0.0

            events = ea.Scalars(reward_tag)
            if not events:
                logger.warning(f"No events found for tag: {reward_tag}")
                return 0.0

            # Enhanced fitness calculation
            values = [event.value for event in events]
            logger.info(f"Found {len(values)} reward values for {run_id}")
            
            if len(values) < 5:
                fitness = values[-1] if values else 0.0
                logger.info(f"Using final value as fitness: {fitness}")
                return fitness
            
            # Use the mean of last 20% of values for stability
            last_portion = max(5, len(values) // 5)
            recent_values = values[-last_portion:]
            
            # Filter out extreme outliers (values beyond 3 standard deviations)
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values)
            
            if std_val > 0:
                filtered_values = [v for v in recent_values if abs(v - mean_val) <= 3 * std_val]
            else:
                filtered_values = recent_values
            
            fitness = np.mean(filtered_values) if filtered_values else mean_val
            logger.info(f"Calculated fitness for {run_id}: {fitness:.4f} (from {len(filtered_values)} filtered values)")
            
            return float(fitness)

        except Exception as e:
            logger.error(f"Error reading logs for {run_id}: {e}")
            logger.error(f"Log directory was: {log_dir}")
            return 0.0


def cleanup_old_runs(self, keep_latest: int = 5):
    """Clean up old training runs to save disk space"""
    try:
        results_dir = Path(self.config.RESULTS_DIR)
        if not results_dir.exists():
            return
            
        # Get all run directories
        run_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("GA_")]
        
        # Sort by modification time (newest first)
        run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the latest N runs
        # for old_run in run_dirs[keep_latest:]:
        #     logger.info(f"Cleaning up old run: {old_run.name}")
        #     import shutil
        #     shutil.rmtree(old_run, ignore_errors=True)
            
    except Exception as e:
        logger.warning(f"Failed to cleanup old runs: {e}")



def estimate_runtime(config: GAConfig) -> str:
    """Estimate total runtime"""
    # Rough estimates based on typical ML-Agents training times
    training_time_per_individual = 15  # minutes (for 50k steps)
    total_evaluations = config.POP_SIZE * config.N_GEN
    
    # Account for parallel processing (if applicable)
    if config.PARALLEL_WORKERS > 1:
        total_evaluations = total_evaluations / min(config.PARALLEL_WORKERS, 2)  # Conservative estimate
    
    total_hours = (total_evaluations * training_time_per_individual) / 60
    
    return f"""
Runtime Estimation:
- Training time per individual: ~{training_time_per_individual} minutes
- Total evaluations: {config.POP_SIZE * config.N_GEN}
- Estimated total time: {total_hours:.1f} hours ({total_hours/24:.1f} days)
- With early stopping: potentially 20-40% less

Note: This is a rough estimate. Actual time depends on:
- Hardware performance
- Unity environment complexity  
- Network architecture
- Convergence speed
"""

if __name__ == "__main__":
    config = GAConfig()
    
    print(estimate_runtime(config))
    
    # Initialize and run GA
    ga = OptimizedGA(config)
    
    try:
        hof, final_pop = ga.run_evolution()
        logger.info("Genetic Algorithm completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Evolution interrupted by user")
    except Exception as e:
        logger.error(f"Evolution failed: {e}")
        raise