# sb3_agent.py
import numpy as np
import torch
from stable_baselines3 import PPO, DQN, A2C, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
import os
from datetime import datetime

class TetrisEnv(gym.Env):
    """
    Gymnasium environment wrapper for Tetris that works with Stable Baselines3
    """
    
    def __init__(self, client, reward_calculator, state_size=208):
        super(TetrisEnv, self).__init__()
        
        self.client = client
        self.reward_calculator = reward_calculator
        self.state_size = state_size
        
        # Define action and observation space
        # Assuming 40 possible actions (as per your DQN setup)
        self.action_space = spaces.Discrete(40)
        
        # State space - flattened board + additional features
        # Adjust this based on your actual state representation
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_size,), dtype=np.float32
        )
        
        self.current_state = None
        self.prev_state = None
        self.step_count = 0
        self.total_steps = 0
        
    def _state_to_observation(self, state):
        # 1) board (200 floats)
        board = np.asarray(state.get('board', [0.]*200), dtype=np.float32)
        if board.shape[0] != 200:
            board = np.resize(board, (200,))  # or pad/truncate as you prefer

        # 2) piece info
        curr = np.asarray(state.get('currentPiece', [0,0,0,0]), dtype=np.float32)
        nxt  = np.asarray(state.get('nextPiece',    [0]),      dtype=np.float32)

        # 3) piece position
        pos  = state.get('piecePosition', {})
        xy   = np.asarray([pos.get('x',0), pos.get('y',0)], dtype=np.float32)

        # 4) simple scalar features
        extras = [
            float(state.get('score', 0)),
            float(state.get('reward', 0.0)),
            float(state.get('holesCount', 0)),
            float(state.get('stackHeight', 0.0)),
            float(state.get('bumpiness', 0.0)),
            # … add whatever else you like …
        ]
        extras = np.asarray(extras, dtype=np.float32)

        # 5) column heights (10 floats)
        heights = np.asarray(state.get('heights', [0]*10), dtype=np.float32)
        if heights.shape[0] != 10:
            heights = np.resize(heights, (10,))

        # concatenate everything
        obs = np.concatenate([board, curr, nxt, xy, extras, heights])
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset the game
        self.client.send_reset()
        
        # Wait for initial state
        initial_state = self.client.wait_for_game_ready(timeout=10.0)
        if initial_state is None:
            initial_state = {'board': [0] * 200, 'score': 0, 'linesCleared': 0}
        
        self.current_state = initial_state
        self.prev_state = None
        self.step_count = 0
        
        observation = self._state_to_observation(initial_state)
        info = {'state': initial_state}
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        self.step_count += 1
        self.total_steps += 1
        
        # Send action to Unity
        next_state = self.client.send_action_and_wait(action, timeout=10.0)
        
        if next_state is None:
            # Handle timeout or connection issues
            reward = -1.0
            terminated = True
            observation = np.zeros(self.state_size, dtype=np.float32)
            info = {'timeout': True}
            return observation, reward, terminated, False, info
        
        # Calculate reward using the existing reward calculator
        reward = self.reward_calculator(self.prev_state, next_state, action, self.total_steps)
        
        # Check if game is over
        terminated = self.client.is_game_over(next_state)
        truncated = False  # We don't use truncation in Tetris
        
        # Convert state to observation
        observation = self._state_to_observation(next_state)
        
        # Update states
        self.prev_state = self.current_state
        self.current_state = next_state
        
        info = {
            'state': next_state,
            'score': next_state.get('score', 0),
            'lines': next_state.get('linesCleared', 0),
            'step_count': self.step_count
        }
        
        return observation, reward, terminated, truncated, info


class TensorBoardCallback(BaseCallback):
    """
    Custom callback for logging metrics to TensorBoard
    """
    
    def __init__(self, log_dir, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_lines = []
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        # Log additional metrics if available
        if hasattr(self.locals, 'infos') and self.locals['infos']:
            for info in self.locals['infos']:
                if 'score' in info:
                    self.logger.record('game/score', info['score'])
                if 'lines' in info:
                    self.logger.record('game/lines', info['lines'])


class StableBaselinesAgent:
    """
    Wrapper for Stable Baselines3 agents to work with the existing trainer
    """
    
    def __init__(
        self, 
        client,
        reward_calculator,
        algorithm='PPO',
        state_size=208,
        tensorboard_log_dir=None,
        learning_rate=3e-4,
        batch_size=64,
        **kwargs
    ):
        self.client = client
        self.reward_calculator = reward_calculator
        self.algorithm = algorithm
        self.state_size = state_size
        self.tensorboard_log_dir = tensorboard_log_dir or f"runs/sb3_{algorithm.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Create environment
        self.env = TetrisEnv(client, reward_calculator, state_size)
        
        # Wrap in DummyVecEnv for SB3 compatibility
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Configure logger
        self.logger = configure(self.tensorboard_log_dir, ["tensorboard"])
        
        # Initialize the agent based on algorithm
        self.model = self._create_model(**kwargs)
        self.model.set_logger(self.logger)
        
        # Create callback
        self.callback = TensorBoardCallback(self.tensorboard_log_dir)
        
        # Memory and other attributes for compatibility
        self.memory = []  # SB3 handles experience replay internally
        self.epsilon = 0.0  # SB3 handles exploration internally
        
    def _create_model(self, **kwargs):
        """Create the SB3 model based on algorithm"""
        
        # Common parameters
        common_params = {
            'policy': 'MlpPolicy',
            'env': self.vec_env,
            'learning_rate': self.learning_rate,
            'verbose': 1,
            'tensorboard_log': self.tensorboard_log_dir,
        }
        
        # Algorithm-specific parameters
        if self.algorithm.upper() == 'PPO':
            model_params = {
                'batch_size': self.batch_size,
                'n_steps': 2048,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                **kwargs
            }
            return PPO(**{**common_params, **model_params})
            
        elif self.algorithm.upper() == 'DQN':
            model_params = {
                'batch_size': self.batch_size,
                'buffer_size': 100000,
                'learning_starts': 1000,
                'target_update_interval': 1000,
                'train_freq': 4,
                'gradient_steps': 1,
                'exploration_fraction': 0.1,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                **kwargs
            }
            return DQN(**{**common_params, **model_params})
            
        elif self.algorithm.upper() == 'A2C':
            model_params = {
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'ent_coef': 0.01,
                'vf_coef': 0.25,
                **kwargs
            }
            return A2C(**{**common_params, **model_params})
            
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def act(self, state, training=True):
        """Choose an action given the current state"""
        observation = self.env._state_to_observation(state)
        observation = observation.reshape(1, -1)  # Add batch dimension
        
        action, _ = self.model.predict(observation, deterministic=not training)
        return int(action[0]) if isinstance(action, np.ndarray) else int(action)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience (not needed for SB3, but kept for compatibility)"""
        pass  # SB3 handles experience collection internally
    
    def replay(self):
        """Train the model (handled internally by SB3)"""
        pass  # SB3 handles training during model.learn()
    
    def learn(self, total_timesteps):
        """Train the model for given timesteps"""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback,
            reset_num_timesteps=False
        )
    
    def save(self, filename):
        """Save the model"""
        self.model.save(filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load the model"""
        if os.path.exists(filename + '.zip'):
            if self.algorithm.upper() == 'PPO':
                self.model = PPO.load(filename, env=self.vec_env)
            elif self.algorithm.upper() == 'DQN':
                self.model = DQN.load(filename, env=self.vec_env)
            elif self.algorithm.upper() == 'A2C':
                self.model = A2C.load(filename, env=self.vec_env)
            
            self.model.set_logger(self.logger)
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file {filename} not found")
    
    def log_episode_metrics(self, episode, episode_reward, steps, score, lines, game_metrics):
        """Log episode metrics (compatible with existing trainer)"""
        self.logger.record('episode/reward', episode_reward)
        self.logger.record('episode/length', steps)
        self.logger.record('episode/score', score)
        self.logger.record('episode/lines', lines)
        
        for key, value in game_metrics.items():
            self.logger.record(f'game/{key}', value)
        
        self.logger.dump(step=episode)
    
    def close(self):
        """Close the agent and clean up"""
        if hasattr(self, 'vec_env'):
            self.vec_env.close()
    
    @property
    def writer(self):
        """Property for compatibility with existing code"""
        return TensorBoardWriterWrapper(self.logger)


# Example usage and integration function
def create_sb3_agent(client, reward_calculator, algorithm='PPO', **kwargs):
    """
    Factory function to create SB3 agent
    """
    return StableBaselinesAgent(
        client=client,
        reward_calculator=reward_calculator,
        algorithm=algorithm,
        **kwargs
    )

class TensorBoardWriterWrapper:
    def __init__(self, sb3_logger):
        self.logger = sb3_logger
        
    def add_scalar(self, tag, scalar_value, global_step):
        """Convert TensorBoard-style calls to SB3 logger calls"""
        self.logger.record(tag, scalar_value)
        # Note: SB3 handles the step internally, so we don't need global_step
    
    def add_hparams(self, hparam_dict, metric_dict):
        """Handle hyperparameter logging"""
        for key, value in hparam_dict.items():
            self.logger.record(f"hparams/{key}", value)
        for key, value in metric_dict.items():
            self.logger.record(f"hparams/{key}", value)
