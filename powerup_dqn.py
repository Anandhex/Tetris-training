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
        self.next_piece = None
        self.game_over = False
        
        # Tetris pieces (simplified - using basic shapes)
        self.pieces = [
            np.array([[1, 1, 1, 1]]),  # I
            np.array([[1, 1], [1, 1]]),  # O
            np.array([[0, 1, 0], [1, 1, 1]]),  # T
            np.array([[1, 1, 0], [0, 1, 1]]),  # S
            np.array([[0, 1, 1], [1, 1, 0]]),  # Z
            np.array([[1, 0, 0], [1, 1, 1]]),  # J
            np.array([[0, 0, 1], [1, 1, 1]]),  # L
        ]
        
        self.piece_positions = [(0, 0)]  # Current piece position
        self.spawn_new_piece()
    
    def spawn_new_piece(self):
        """Spawn a new piece at the top"""
        self.current_piece = random.choice(self.pieces)
        self.piece_positions = [(0, self.width // 2 - 1)]
        
        # Check if game over
        if self.check_collision(self.current_piece, self.piece_positions[0]):
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
            
        new_pos = (self.piece_positions[0][0] + 1, self.piece_positions[0][1])
        if not self.check_collision(self.current_piece, new_pos):
            self.piece_positions[0] = new_pos
            return True
        else:
            # Place piece and spawn new one
            self.place_piece(self.current_piece, self.piece_positions[0])
            self.clear_lines()
            self.spawn_new_piece()
            return False
    
    def get_board_state(self):
        """Get current board state including current piece"""
        state = self.board.copy()
        if self.current_piece is not None:
            py, px = self.piece_positions[0]
            for y in range(self.current_piece.shape[0]):
                for x in range(self.current_piece.shape[1]):
                    if self.current_piece[y, x] == 1:
                        new_y, new_x = py + y, px + x
                        if 0 <= new_y < self.height and 0 <= new_x < self.width:
                            state[new_y, new_x] = 2  # Current piece
        return state
    
    def apply_powerup(self, powerup_type, position=None):
        """Apply powerup effect"""
        if powerup_type == PowerUpType.BOTTOM_LINE_CLEAR:
            # Clear bottom line
            if np.any(self.board[-1] == 1):
                self.board = np.delete(self.board, -1, axis=0)
                self.board = np.vstack([np.zeros((1, self.width)), self.board])
                self.score += 50
                return True
            return False
        
        elif powerup_type == PowerUpType.GRAVITY:
            # Apply gravity - move all pieces down to fill holes
            for x in range(self.width):
                column = self.board[:, x]
                filled_blocks = column[column == 1]
                empty_blocks = np.zeros(len(column) - len(filled_blocks))
                self.board[:, x] = np.concatenate([empty_blocks, filled_blocks])
            return True
        
        elif powerup_type == PowerUpType.BOMB:
            # Destroy 3x3 area around position
            if position is None:
                # Random position if not specified
                position = (random.randint(1, self.height-2), random.randint(1, self.width-2))
            
            py, px = position
            destroyed = False
            for y in range(max(0, py-1), min(self.height, py+2)):
                for x in range(max(0, px-1), min(self.width, px+2)):
                    if self.board[y, x] == 1:
                        self.board[y, x] = 0
                        destroyed = True
            
            if destroyed:
                self.score += 30
                return True
            return False
        
        return False
    
    def get_features(self):
        """Extract features for ML model"""
        features = []
        
        # Board density
        features.append(np.sum(self.board) / (self.width * self.height))
        
        # Height of each column
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
        
        # Holes count
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
        
        # Bumpiness (height differences between adjacent columns)
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        features.append(bumpiness)
        
        # Lines that can be cleared
        complete_lines = 0
        for y in range(self.height):
            if np.all(self.board[y] == 1):
                complete_lines += 1
        features.append(complete_lines)
        
        # Score and lines cleared
        features.append(self.score / 1000)  # Normalized
        features.append(self.lines_cleared)
        
        return np.array(features, dtype=np.float32)

class PowerUpDQN(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(PowerUpDQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, 4)  # 4 actions: use powerup or not (for 3 types + no action)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class PowerUpAgent:
    def __init__(self, state_size, lr=0.001):
        self.state_size = state_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = PowerUpDQN(state_size).to(self.device)
        self.target_network = PowerUpDQN(state_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, available_powerup=None):
        """Choose action using epsilon-greedy policy"""
        if available_powerup is None:
            return 3  # No action
        
        if random.random() <= self.epsilon:
            return random.randrange(4)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size=32):
        """Train the network on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class TetrisEnvironment:
    def __init__(self):
        self.board = TetrisBoard()
        self.powerup_threshold = 500  # Score threshold for powerup
        self.current_powerup = None
        self.powerup_used = False
        
    def reset(self):
        """Reset environment"""
        self.board = TetrisBoard()
        self.current_powerup = None
        self.powerup_used = False
        return self.board.get_features()
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        reward = 0
        done = False
        
        # Check if powerup should be given
        if self.board.score >= self.powerup_threshold and self.current_powerup is None:
            self.current_powerup = random.choice(list(PowerUpType))
            self.powerup_used = False
        
        # Handle powerup action
        if self.current_powerup is not None and not self.powerup_used:
            if action == self.current_powerup.value:
                # Use powerup
                powerup_reward = self.calculate_powerup_reward()
                if self.board.apply_powerup(self.current_powerup):
                    reward += powerup_reward
                else:
                    reward -= 10  # Penalty for using powerup when not beneficial
                self.powerup_used = True
                self.current_powerup = None
            elif action == 3:  # Don't use powerup
                reward += self.calculate_no_powerup_reward()
                self.current_powerup = None
        
        # Simulate game progress
        old_score = self.board.score
        self.board.move_piece_down()
        
        # Reward for score increase
        score_diff = self.board.score - old_score
        reward += score_diff * 0.1
        
        # Check if game over
        if self.board.game_over:
            done = True
            reward -= 100
        
        return self.board.get_features(), reward, done
    
    def calculate_powerup_reward(self):
        """Calculate reward for using powerup based on board state"""
        features = self.board.get_features()
        
        if self.current_powerup == PowerUpType.BOTTOM_LINE_CLEAR:
            # Reward based on bottom line fullness
            bottom_line_density = np.sum(self.board.board[-1]) / self.board.width
            return bottom_line_density * 50
        
        elif self.current_powerup == PowerUpType.GRAVITY:
            # Reward based on number of holes
            holes = features[12]  # Holes feature
            return holes * 20
        
        elif self.current_powerup == PowerUpType.BOMB:
            # Reward based on board density
            density = features[0]  # Board density
            return density * 30
        
        return 0
    
    def calculate_no_powerup_reward(self):
        """Calculate reward for not using powerup"""
        features = self.board.get_features()
        
        if self.current_powerup == PowerUpType.BOTTOM_LINE_CLEAR:
            # Small penalty if bottom line is very full
            bottom_line_density = np.sum(self.board.board[-1]) / self.board.width
            return -bottom_line_density * 10
        
        elif self.current_powerup == PowerUpType.GRAVITY:
            # Small penalty if many holes
            holes = features[12]
            return -holes * 5
        
        elif self.current_powerup == PowerUpType.BOMB:
            # Small penalty if board is very dense
            density = features[0]
            return -density * 5
        
        return 5  # Small reward for saving powerup

def train_powerup_agent(episodes=1000):
    """Train the powerup agent"""
    env = TetrisEnvironment()
    state_size = len(env.board.get_features())
    agent = PowerUpAgent(state_size)
    
    scores = []
    powerup_usage = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        powerups_used = 0
        steps = 0
        
        while not env.board.game_over and steps < 1000:
            action = agent.act(state, env.current_powerup)
            next_state, reward, done = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if env.current_powerup is not None and action != 3:
                powerups_used += 1
            
            steps += 1
            
            if done:
                break
        
        scores.append(env.board.score)
        powerup_usage.append(powerups_used)
        
        # Train the agent
        if len(agent.memory) > 32:
            agent.replay(32)
        
        # Update target network every 100 episodes
        if episode % 100 == 0:
            agent.update_target_network()
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_powerup_usage = np.mean(powerup_usage[-100:])
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, "
                  f"Avg Powerup Usage: {avg_powerup_usage:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return agent, scores, powerup_usage

def evaluate_agent(agent, episodes=10):
    """Evaluate trained agent"""
    env = TetrisEnvironment()
    scores = []
    
    agent.epsilon = 0  # No exploration during evaluation
    
    for episode in range(episodes):
        state = env.reset()
        steps = 0
        
        while not env.board.game_over and steps < 1000:
            action = agent.act(state, env.current_powerup)
            state, _, done = env.step(action)
            steps += 1
            
            if done:
                break
        
        scores.append(env.board.score)
        print(f"Evaluation Episode {episode + 1}: Score = {env.board.score}")
    
    return scores

def visualize_training_results(scores, powerup_usage):
    """Visualize training results"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(powerup_usage)
    plt.title('Powerup Usage')
    plt.xlabel('Episode')
    plt.ylabel('Powerups Used')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Training PowerUp DQN Agent...")
    agent, scores, powerup_usage = train_powerup_agent(episodes=1000)
    
    print("\nTraining completed!")
    print(f"Average score (last 100 episodes): {np.mean(scores[-100:]):.2f}")
    print(f"Average powerup usage (last 100 episodes): {np.mean(powerup_usage[-100:]):.2f}")
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_scores = evaluate_agent(agent, episodes=10)
    print(f"Average evaluation score: {np.mean(eval_scores):.2f}")
    
    # Visualize results
    visualize_training_results(scores, powerup_usage)
    
    # Save the trained model
    torch.save(agent.q_network.state_dict(), 'powerup_dqn_model.pth')
    print("\nModel saved as 'powerup_dqn_model.pth'")