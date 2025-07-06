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
        self.pieces_placed = 0
        
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
        
        # Create some initial blocks for testing powerups
        self.create_initial_blocks()
    
    def create_initial_blocks(self):
        """Create some initial blocks to make powerups more effective"""
        # Add some scattered blocks
        for _ in range(10):
            x = random.randint(0, self.width - 1)
            y = random.randint(self.height - 8, self.height - 1)
            self.board[y, x] = 1
        
        # Add some holes
        for _ in range(3):
            x = random.randint(0, self.width - 1)
            y = random.randint(self.height - 5, self.height - 1)
            self.board[y, x] = 0
        
        # DEBUG: Print initial board
        print("\nINITIAL BOARD STATE:")
        for row in self.board:
            print(' '.join('X' if cell else '.' for cell in row))
        print(f"Density: {np.sum(self.board)/(self.width*self.height):.2f}")
    
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
                    # Check boundaries and existing blocks
                    if (new_y >= self.height or new_x < 0 or new_x >= self.width or
                        (new_y >= 0 and self.board[new_y, new_x] != 0)):
                        return True # Collision detected
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
        self.pieces_placed += 1
    
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
        self.score += lines_cleared * 100 + self.pieces_placed * 10  # Score for pieces + lines
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
        """Apply powerup effect and return effectiveness score"""
        effectiveness = 0
        
        if powerup_type == PowerUpType.BOTTOM_LINE_CLEAR:
            # Clear bottom line if it has blocks
            bottom_blocks = np.sum(self.board[-1])
            if bottom_blocks > 0:
                self.board = np.delete(self.board, -1, axis=0)
                self.board = np.vstack([np.zeros((1, self.width)), self.board])
                effectiveness = bottom_blocks * 10
                self.score += effectiveness
                print(f"Bottom line cleared! Removed {bottom_blocks} blocks, score +{effectiveness}")
                return effectiveness
            else:
                print("Bottom line clear - no blocks to remove")
                return 0
        
        elif powerup_type == PowerUpType.GRAVITY:
            # Apply gravity to fill holes
            blocks_moved = 0
            for x in range(self.width):
                column = self.board[:, x]
                filled_blocks = column[column == 1]
                empty_count = len(column) - len(filled_blocks)
                
                if empty_count > 0 and len(filled_blocks) > 0:
                    # Count how many blocks will move
                    for y in range(len(column)):
                        if column[y] == 0:
                            # Check if there are blocks above
                            if np.any(column[y:] == 1):
                                blocks_moved += 1
                                break
                    
                    # Apply gravity
                    empty_blocks = np.zeros(empty_count)
                    self.board[:, x] = np.concatenate([empty_blocks, filled_blocks])
            
            if blocks_moved > 0:
                effectiveness = blocks_moved * 5
                self.score += effectiveness
                print(f"Gravity applied! Moved {blocks_moved} blocks, score +{effectiveness}")
                return effectiveness
            else:
                print("Gravity - no blocks to move")
                return 0
        
        elif powerup_type == PowerUpType.BOMB:
            # Bomb destroys 3x3 area
            if position is None:
                # Find a good position with blocks
                best_pos = None
                max_blocks = 0
                for py in range(1, self.height - 1):
                    for px in range(1, self.width - 1):
                        block_count = 0
                        for y in range(py - 1, py + 2):
                            for x in range(px - 1, px + 2):
                                if self.board[y, x] == 1:
                                    block_count += 1
                        if block_count > max_blocks:
                            max_blocks = block_count
                            best_pos = (py, px)
                position = best_pos if best_pos else (self.height // 2, self.width // 2)
            
            py, px = position
            destroyed = 0
            for y in range(max(0, py - 1), min(self.height, py + 2)):
                for x in range(max(0, px - 1), min(self.width, px + 2)):
                    if self.board[y, x] == 1:
                        self.board[y, x] = 0
                        destroyed += 1
            
            if destroyed > 0:
                effectiveness = destroyed * 8
                self.score += effectiveness
                print(f"Bomb exploded at ({py},{px})! Destroyed {destroyed} blocks, score +{effectiveness}")
                return effectiveness
            else:
                print("Bomb - no blocks destroyed")
                return 0
        
        return 0
    
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

# DQN Neural Network
class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, 4)  # 4 actions: use powerup 0, 1, 2, or don't use (3)
        
        self.init_weights()
    
    def init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size=4, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.gamma = 0.95
        self.batch_size = 32
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQNNetwork(state_size).to(self.device)
        self.target_network = DQNNetwork(state_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
        self.update_target_network()
        self.losses = []
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, available_powerup=None):
        if available_powerup is None:
            return 3  # No powerup available
        
        if random.random() <= self.epsilon:
            # Random action: either use the available powerup or don't use it
            return random.choice([available_powerup.value, 3])
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        self.q_network.train()
        
        return q_values.argmax().item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (~dones))
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class TetrisEnvironment:
    def __init__(self):
        self.board = TetrisBoard()
        self.powerup_threshold = 20  # Lower threshold for more frequent powerups
        self.powerup_interval = 5   # Give powerup every 20 pieces
        self.current_powerup = None
        self.powerup_available = False
        self.steps = 0
        self.max_steps = 500
        self.powerup_decisions = 0
        
    def reset(self):
        self.board = TetrisBoard()
        self.current_powerup = None
        self.powerup_available = False
        self.steps = 0
        self.powerup_decisions = 0
        return self.board.get_features()
    
    def step(self, action):
        self.steps += 1
        reward = 0
        done = False

        print(f"🔍 Powerup check: Pieces={self.board.pieces_placed} "
        f"(Need {self.powerup_interval}) | "
        f"Score={self.board.score} (Need {self.powerup_threshold})")
        
        # **POWERUP ASSIGNMENT LOGIC**
        # Give powerup every 'powerup_interval' pieces placed OR when score reaches threshold
        if (self.board.pieces_placed > 0 and 
            self.board.pieces_placed % self.powerup_interval == 0 and 
            not self.powerup_available):
            
            self.current_powerup = random.choice(list(PowerUpType))
            self.powerup_available = True
            print(f"\n🎁 POWERUP ASSIGNED: {self.current_powerup.name} at piece {self.board.pieces_placed}")
        
        # **POWERUP DECISION HANDLING**
        if self.powerup_available and self.current_powerup is not None:
            self.powerup_decisions += 1
            
            if action == self.current_powerup.value:
                # Agent chose to USE the powerup
                old_score = self.board.score
                effectiveness = self.board.apply_powerup(self.current_powerup)
                
                if effectiveness > 0:
                    reward += effectiveness * 2  # Reward for effective use
                    print(f"✅ POWERUP USED effectively! Reward: +{effectiveness * 2}")
                else:
                    reward -= 20  # Penalty for ineffective use
                    print(f"❌ POWERUP USED ineffectively! Penalty: -20")
                
                # Powerup consumed
                self.powerup_available = False
                self.current_powerup = None
                
            elif action == 3:
                # Agent chose NOT to use powerup
                reward += self.evaluate_not_using_powerup()
                print(f"🤔 POWERUP NOT USED. Reward: +{self.evaluate_not_using_powerup()}")
                
                # Powerup consumed (decision made)
                self.powerup_available = False
                self.current_powerup = None
            
            else:
                # Agent chose wrong action (shouldn't happen with proper logic)
                reward -= 10
                print(f"⚠️ WRONG ACTION: chose {action} for powerup {self.current_powerup.value}")
        
        # **GAME SIMULATION**
        old_score = self.board.score
        self.board.move_piece_down()
        
        # Reward for score increase
        score_diff = self.board.score - old_score
        reward += score_diff * 0.1
        
        # Small step penalty to encourage efficiency
        reward -= 1
        
        # **TERMINATION CONDITIONS**
        if self.board.game_over:
            done = True
            reward -= 100
            print(f"💀 GAME OVER! Final score: {self.board.score}")
        elif self.steps >= self.max_steps:
            done = True
            print(f"⏰ MAX STEPS REACHED! Final score: {self.board.score}")
        
        return self.board.get_features(), reward, done
    
    def evaluate_not_using_powerup(self):
        """Evaluate whether NOT using powerup was a good decision"""
        if self.current_powerup == PowerUpType.BOTTOM_LINE_CLEAR:
            bottom_density = np.sum(self.board.board[-1]) / self.board.width
            return 10 if bottom_density < 0.3 else -10
        
        elif self.current_powerup == PowerUpType.GRAVITY:
            features = self.board.get_features()
            holes = features[11]
            return 10 if holes < 2 else -10
        
        elif self.current_powerup == PowerUpType.BOMB:
            density = features[0] if 'features' in locals() else self.board.get_features()[0]
            return 10 if density < 0.3 else -10
        
        return 5

def train_dqn_agent(episodes=500):
    """Train DQN agent with detailed logging"""
    print("🚀 Starting DQN Training...")
    print("=" * 60)
    
    env = TetrisEnvironment()
    state_size = len(env.board.get_features())
    agent = DQNAgent(state_size)
    
    scores = []
    powerup_decisions = []
    episode_rewards = []
    
    print(f"📊 Training Configuration:")
    print(f"   State size: {state_size}")
    print(f"   Action size: {agent.action_size}")
    print(f"   Episodes: {episodes}")
    print(f"   Device: {agent.device}")
    print("=" * 60)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        powerup_count = 0
        
        print(f"\n🎮 Episode {episode + 1}/{episodes}")
        print(f"   Starting score: {env.board.score}")
        
        while True:
            # Show current state
            if env.powerup_available:
                print(f"🎁 POWERUP AVAILABLE! {env.current_powerup.name} | "
                    f"Pieces: {env.board.pieces_placed} | "
                    f"Score: {env.board.score}")
            
            # Agent makes decision
            action = agent.act(state, env.current_powerup if env.powerup_available else None)
            
            # Environment step
            next_state, reward, done = env.step(action)
            
            # Store experience for training
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if env.powerup_decisions > powerup_count:
                powerup_count = env.powerup_decisions
            
            # Train neural network
            agent.replay()
            
            if done:
                break
        
        # Episode summary
        scores.append(env.board.score)
        powerup_decisions.append(powerup_count)
        episode_rewards.append(total_reward)
        
        print(f"   📊 Episode {episode + 1} Summary:")
        print(f"      Final score: {env.board.score}")
        print(f"      Total reward: {total_reward:.2f}")
        print(f"      Powerup decisions: {powerup_count}")
        print(f"      Steps: {env.steps}")
        print(f"      Epsilon: {agent.epsilon:.3f}")
        print(f"      Memory size: {len(agent.memory)}")
        
        # Update target network every 50 episodes
        if episode % 50 == 0:
            agent.update_target_network()
            print(f"   🎯 Target network updated!")
        
        # Progress report every 25 episodes
        if episode % 25 == 0 and episode > 0:
            recent_scores = scores[-25:]
            recent_rewards = episode_rewards[-25:]
            recent_powerups = powerup_decisions[-25:]
            
            print(f"\n📈 Progress Report (Episodes {episode-24}-{episode+1}):")
            print(f"   Average score: {np.mean(recent_scores):.2f}")
            print(f"   Average reward: {np.mean(recent_rewards):.2f}")
            print(f"   Average powerup decisions: {np.mean(recent_powerups):.2f}")
            print(f"   Average loss: {np.mean(agent.losses[-100:]) if len(agent.losses) > 0 else 0:.4f}")
    
    print("\n🎉 Training completed!")
    return agent, scores, powerup_decisions, episode_rewards

def test_powerups_individually():
    """Test each powerup individually to verify they work"""
    print("🔧 Testing Powerups Individually...")
    print("=" * 50)
    
    for powerup in PowerUpType:
        print(f"\n🧪 Testing {powerup.name}:")
        board = TetrisBoard()
        
        print(f"   Before: Score = {board.score}")
        print(f"   Board density: {np.sum(board.board)/(board.width * board.height):.2f}")
        
        effectiveness = board.apply_powerup(powerup)
        
        print(f"   After: Score = {board.score}")
        print(f"   Effectiveness: {effectiveness}")
        print(f"   New density: {np.sum(board.board)/(board.width * board.height):.2f}")

if __name__ == "__main__":
    # First test powerups
    test_powerups_individually()
    
    # Then train
    agent, scores, powerup_decisions, episode_rewards = train_dqn_agent(episodes=200)
    
    # Results
    print(f"\n📊 Final Results:")
    print(f"   Average score (last 50): {np.mean(scores[-50:]):.2f}")
    print(f"   Average powerup decisions (last 50): {np.mean(powerup_decisions[-50:]):.2f}")
    print(f"   Best score: {max(scores)}")
    print(f"   Total powerup decisions: {sum(powerup_decisions)}")
    
    # Save model
    torch.save(agent.q_network.state_dict(), 'tetris_powerup_dqn.pth')
    print(f"\n💾 Model saved as 'tetris_powerup_dqn.pth'")