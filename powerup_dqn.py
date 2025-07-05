import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from enum import Enum
import copy

class PowerUpType(Enum):
    BOTTOM_LINE_CLEAR = 0
    GRAVITY = 1
    BOMB = 2

class TetrisBoard:
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.board = np.zeros((height, width), dtype=int)
        self.score = 0
        self.lines_cleared = 0
        self.current_piece = None
        self.game_over = False
        
        # Simplified Tetris pieces
        self.pieces = [
            np.array([[1, 1, 1, 1]]),  # I
            np.array([[1, 1], [1, 1]]),  # O
            np.array([[0, 1, 0], [1, 1, 1]]),  # T
            np.array([[1, 1, 0], [0, 1, 1]]),  # S
            np.array([[0, 1, 1], [1, 1, 0]]),  # Z
            np.array([[1, 0, 0], [1, 1, 1]]),  # J
            np.array([[0, 0, 1], [1, 1, 1]]),  # L
        ]
        
        self.piece_position = (0, 0)
        self.spawn_new_piece()
    
    def spawn_new_piece(self):
        """Spawn a new piece at the top"""
        self.current_piece = random.choice(self.pieces)
        self.piece_position = (0, self.width // 2 - 1)
        
        # Check if game over
        if self.check_collision(self.current_piece, self.piece_position):
            self.game_over = True
    
    def check_collision(self, piece, position):
        """Check if piece collides with board or boundaries"""
        py, px = position
        for y in range(piece.shape[0]):
            for x in range(piece.shape[1]):
                if piece[y, x] == 1:
                    new_y, new_x = py + y, px + x
                    if (new_y >= self.height or new_x < 0 or new_x >= self.width or
                        (new_y >= 0 and self.board[new_y, new_x] != 0)):
                        return True
        return False
    
    def place_piece(self, piece, position):
        """Place piece on board"""
        py, px = position
        for y in range(piece.shape[0]):
            for x in range(piece.shape[1]):
                if piece[y, x] == 1:
                    new_y, new_x = py + y, px + x
                    if 0 <= new_y < self.height and 0 <= new_x < self.width:
                        self.board[new_y, new_x] = 1
    
    def clear_lines(self):
        """Clear completed lines and return number of lines cleared"""
        lines_to_clear = []
        for y in range(self.height):
            if np.all(self.board[y] == 1):
                lines_to_clear.append(y)
        
        for y in reversed(lines_to_clear):
            self.board = np.delete(self.board, y, axis=0)
            self.board = np.vstack([np.zeros((1, self.width)), self.board])
        
        lines_cleared = len(lines_to_clear)
        self.lines_cleared += lines_cleared
        self.score += lines_cleared * 100
        return lines_cleared
    
    def move_piece_down(self):
        """Move current piece down, return True if successful"""
        if self.current_piece is None:
            return False
            
        new_pos = (self.piece_position[0] + 1, self.piece_position[1])
        if not self.check_collision(self.current_piece, new_pos):
            self.piece_position = new_pos
            return True
        else:
            # Place piece and spawn new one
            self.place_piece(self.current_piece, self.piece_position)
            self.clear_lines()
            self.spawn_new_piece()
            return False
    
    def apply_powerup(self, powerup_type, position=None):
        """Apply powerup effect"""
        if powerup_type == PowerUpType.BOTTOM_LINE_CLEAR:
            if np.any(self.board[-1] == 1):
                self.board = np.delete(self.board, -1, axis=0)
                self.board = np.vstack([np.zeros((1, self.width)), self.board])
                self.score += 50
                return True
            return False
        
        elif powerup_type == PowerUpType.GRAVITY:
            applied = False
            for x in range(self.width):
                column = self.board[:, x]
                filled_blocks = column[column == 1]
                if len(filled_blocks) < len(column):
                    empty_blocks = np.zeros(len(column) - len(filled_blocks))
                    self.board[:, x] = np.concatenate([empty_blocks, filled_blocks])
                    applied = True
            if applied:
                self.score += 30
            return applied
        
        elif powerup_type == PowerUpType.BOMB:
            if position is None:
                position = (random.randint(1, self.height-2), random.randint(1, self.width-2))
            
            py, px = position
            destroyed = False
            for y in range(max(0, py-1), min(self.height, py+2)):
                for x in range(max(0, px-1), min(self.width, px+2)):
                    if self.board[y, x] == 1:
                        self.board[y, x] = 0
                        destroyed = True
            
            if destroyed:
                self.score += 40
                return True
            return False
        
        return False
    
    def get_features(self):
        """Extract features for ML model"""
        features = []
        
        # 1. Board density
        features.append(np.sum(self.board) / (self.width * self.height))
        
        # 2. Height of each column
        heights = []
        for x in range(self.width):
            column = self.board[:, x]
            height = 0
            for y in range(self.height):
                if column[y] == 1:
                    height = self.height - y
                    break
            heights.append(height)
        
        features.extend(heights)
        
        # 3. Number of holes
        holes = 0
        for x in range(self.width):
            column = self.board[:, x]
            found_block = False
            for y in range(self.height):
                if column[y] == 1:
                    found_block = True
                elif found_block and column[y] == 0:
                    holes += 1
        features.append(holes)
        
        # 4. Bumpiness (height differences between adjacent columns)
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        features.append(bumpiness)
        
        # 5. Maximum height
        features.append(max(heights))
        
        # 6. Number of complete lines
        complete_lines = 0
        for y in range(self.height):
            if np.all(self.board[y] == 1):
                complete_lines += 1
        features.append(complete_lines)
        
        # 7. Bottom line density
        bottom_line_density = np.sum(self.board[-1]) / self.width
        features.append(bottom_line_density)
        
        # 8. Score (normalized)
        features.append(self.score / 1000)
        
        return np.array(features, dtype=np.float32)

# CORE DQN NEURAL NETWORK
class DQNNetwork(nn.Module):
    """Deep Q-Network for powerup decision making"""
    def __init__(self, input_size, hidden_size=256):
        super(DQNNetwork, self).__init__()
        
        # Define the neural network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 4)  # 4 actions: use powerup 0, 1, 2, or don't use (3)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize network weights"""
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)  # No activation on output layer
        return x

class DQNAgent:
    """DQN Agent with complete neural network training"""
    def __init__(self, state_size, action_size=4, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Experience replay memory
        self.memory = deque(maxlen=50000)
        self.batch_size = 32
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Neural Networks
        self.q_network_local = DQNNetwork(state_size).to(self.device)
        self.q_network_target = DQNNetwork(state_size).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=lr)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Initialize target network
        self.hard_update(self.q_network_local, self.q_network_target)
        
        # Training metrics
        self.losses = []
        self.q_values = []
        
    def hard_update(self, local_model, target_model):
        """Copy weights from local to target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
    
    def soft_update(self, local_model, target_model, tau=0.001):
        """Soft update target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, available_powerup=None):
        """Choose action using epsilon-greedy policy"""
        if available_powerup is None:
            return 3  # No powerup available, don't use
        
        # Epsilon-greedy action selection
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Get Q-values from neural network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network_local.eval()
        with torch.no_grad():
            q_values = self.q_network_local(state_tensor)
        self.q_network_local.train()
        
        # Store Q-values for analysis
        self.q_values.append(q_values.cpu().numpy().flatten())
        
        return q_values.argmax().item()
    
    def replay(self):
        """CORE DQN TRAINING: Experience replay with neural network training"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Get current Q-values for chosen actions
        current_q_values = self.q_network_local(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q-values from target network
        next_q_values = self.q_network_target(next_states).max(1)[0].detach()
        
        # Compute target Q-values using Bellman equation
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # NEURAL NETWORK TRAINING STEP
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backward propagation
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        
        self.optimizer.step()  # Update weights
        
        # Store loss for analysis
        self.losses.append(loss.item())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network"""
        self.hard_update(self.q_network_local, self.q_network_target)

class TetrisEnvironment:
    """Tetris environment for training DQN"""
    def __init__(self):
        self.board = TetrisBoard()
        self.powerup_threshold = 300  # Score threshold for powerup
        self.current_powerup = None
        self.powerup_used = False
        self.steps = 0
        self.max_steps = 500
        
    def reset(self):
        """Reset environment"""
        self.board = TetrisBoard()
        self.current_powerup = None
        self.powerup_used = False
        self.steps = 0
        return self.board.get_features()
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        self.steps += 1
        reward = 0
        done = False
        
        # Check if powerup should be assigned
        if (self.board.score >= self.powerup_threshold and 
            self.current_powerup is None and 
            not self.powerup_used):
            self.current_powerup = random.choice(list(PowerUpType))
            self.powerup_used = False
        
        # Handle powerup decision
        if self.current_powerup is not None and not self.powerup_used:
            if action == self.current_powerup.value:
                # Agent chose to use the powerup
                old_score = self.board.score
                powerup_effective = self.board.apply_powerup(self.current_powerup)
                
                if powerup_effective:
                    # Powerup was effective
                    reward += 20 + (self.board.score - old_score)
                else:
                    # Powerup was not effective
                    reward -= 10
                
                self.powerup_used = True
                self.current_powerup = None
                
            elif action == 3:  # Don't use powerup
                # Agent chose not to use powerup
                reward += self.evaluate_not_using_powerup()
                self.current_powerup = None
            else:
                # Agent chose wrong powerup type
                reward -= 5
                self.current_powerup = None
        
        # Simulate game progression
        old_score = self.board.score
        self.board.move_piece_down()
        
        # Reward for score increase
        score_diff = self.board.score - old_score
        reward += score_diff * 0.1
        
        # Small negative reward for each step (encourages efficiency)
        reward -= 0.1
        
        # Check termination conditions
        if self.board.game_over:
            done = True
            reward -= 50  # Penalty for game over
        elif self.steps >= self.max_steps:
            done = True
        
        return self.board.get_features(), reward, done
    
    def evaluate_not_using_powerup(self):
        """Evaluate reward for not using powerup"""
        if self.current_powerup == PowerUpType.BOTTOM_LINE_CLEAR:
            bottom_density = np.sum(self.board.board[-1]) / self.board.width
            return 5 if bottom_density < 0.5 else -5
        
        elif self.current_powerup == PowerUpType.GRAVITY:
            features = self.board.get_features()
            holes = features[11]  # Holes feature
            return 5 if holes < 3 else -5
        
        elif self.current_powerup == PowerUpType.BOMB:
            density = self.board.get_features()[0]
            return 5 if density < 0.3 else -5
        
        return 0

def train_dqn_agent(episodes=2000, target_update=100):
    """Complete DQN training with neural network"""
    print("Initializing DQN training...")
    
    # Initialize environment and agent
    env = TetrisEnvironment()
    state_size = len(env.board.get_features())
    agent = DQNAgent(state_size)
    
    # Training metrics
    scores = []
    episode_lengths = []
    powerup_decisions = []
    avg_losses = []
    
    print(f"State size: {state_size}")
    print(f"Action size: {agent.action_size}")
    print(f"Using device: {agent.device}")
    print("Starting training...\n")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        powerup_count = 0
        
        while True:
            # Agent selects action
            action = agent.act(state, env.current_powerup)
            
            # Environment step
            next_state, reward, done = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            
            if env.current_powerup is not None:
                powerup_count += 1
            
            # Train the neural network
            agent.replay()
            
            if done:
                break
        
        # Update target network periodically
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Store metrics
        scores.append(env.board.score)
        episode_lengths.append(env.steps)
        powerup_decisions.append(powerup_count)
        
        # Calculate average loss
        if len(agent.losses) > 0:
            recent_losses = agent.losses[-env.steps:] if len(agent.losses) >= env.steps else agent.losses
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            avg_losses.append(avg_loss)
        else:
            avg_losses.append(0)
        
        # Print progress
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_powerup = np.mean(powerup_decisions[-100:])
            recent_avg_loss = np.mean(avg_losses[-100:]) if avg_losses else 0
            
            print(f"Episode {episode:4d} | "
                  f"Avg Score: {avg_score:6.1f} | "
                  f"Avg Length: {avg_length:4.1f} | "
                  f"Avg Powerups: {avg_powerup:3.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {recent_avg_loss:.4f} | "
                  f"Memory: {len(agent.memory)}")
    
    print("\nTraining completed!")
    return agent, scores, episode_lengths, powerup_decisions, avg_losses

def evaluate_trained_agent(agent, episodes=20):
    """Evaluate the trained DQN agent"""
    print("\nEvaluating trained agent...")
    
    env = TetrisEnvironment()
    agent.epsilon = 0  # No exploration during evaluation
    
    eval_scores = []
    eval_powerup_usage = []
    
    for episode in range(episodes):
        state = env.reset()
        powerup_used = 0
        
        while True:
            action = agent.act(state, env.current_powerup)
            state, reward, done = env.step(action)
            
            if env.current_powerup is not None and action != 3:
                powerup_used += 1
            
            if done:
                break
        
        eval_scores.append(env.board.score)
        eval_powerup_usage.append(powerup_used)
        
        print(f"Eval Episode {episode+1:2d}: Score = {env.board.score:4d}, "
              f"Powerups Used: {powerup_used}, Steps: {env.steps}")
    
    print(f"\nEvaluation Results:")
    print(f"Average Score: {np.mean(eval_scores):.2f}")
    print(f"Average Powerup Usage: {np.mean(eval_powerup_usage):.2f}")
    print(f"Best Score: {max(eval_scores)}")
    
    return eval_scores

def plot_training_results(scores, losses, powerup_decisions):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Scores
    axes[0, 0].plot(scores)
    axes[0, 0].set_title('Training Scores')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True)
    
    # Moving average of scores
    window = 100
    moving_avg = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
    axes[0, 1].plot(moving_avg)
    axes[0, 1].set_title(f'Moving Average Scores (window={window})')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Average Score')
    axes[0, 1].grid(True)
    
    # Losses
    axes[1, 0].plot(losses)
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Powerup decisions
    axes[1, 1].plot(powerup_decisions)
    axes[1, 1].set_title('Powerup Decisions per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Powerup Decisions')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Train the DQN agent
    agent, scores, episode_lengths, powerup_decisions, avg_losses = train_dqn_agent(episodes=1000)
    
    # Evaluate the trained agent
    eval_scores = evaluate_trained_agent(agent)
    
    # Plot results
    plot_training_results(scores, avg_losses, powerup_decisions)
    
    # Save the trained model
    torch.save({
        'model_state_dict': agent.q_network_local.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'scores': scores,
        'losses': avg_losses
    }, 'dqn_powerup_model.pth')
    
    print("\nModel saved as 'dqn_powerup_model.pth'")
    print(f"Final training score average: {np.mean(scores[-100:]):.2f}")
    print(f"Final evaluation score average: {np.mean(eval_scores):.2f}")