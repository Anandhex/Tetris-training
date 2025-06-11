import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import numpy as np
import os
from datetime import datetime

class TetrisPiece:
    """Tetris piece definitions with rotations and shapes"""
    
    PIECES = {
        'I': [
            [[1, 1, 1, 1]],  # Horizontal
            [[1], [1], [1], [1]]  # Vertical
        ],
        'O': [
            [[1, 1], [1, 1]]  # Square (no rotation)
        ],
        'T': [
            [[0, 1, 0], [1, 1, 1]],  # Normal
            [[1, 0], [1, 1], [1, 0]],  # Right
            [[1, 1, 1], [0, 1, 0]],  # Upside down
            [[0, 1], [1, 1], [0, 1]]  # Left
        ],
        'S': [
            [[0, 1, 1], [1, 1, 0]],  # Normal
            [[1, 0], [1, 1], [0, 1]]  # Vertical
        ],
        'Z': [
            [[1, 1, 0], [0, 1, 1]],  # Normal
            [[0, 1], [1, 1], [1, 0]]  # Vertical
        ],
        'J': [
            [[1, 0, 0], [1, 1, 1]],  # Normal
            [[1, 1], [1, 0], [1, 0]],  # Right
            [[1, 1, 1], [0, 0, 1]],  # Upside down
            [[0, 1], [0, 1], [1, 1]]  # Left
        ],
        'L': [
            [[0, 0, 1], [1, 1, 1]],  # Normal
            [[1, 0], [1, 0], [1, 1]],  # Right
            [[1, 1, 1], [1, 0, 0]],  # Upside down
            [[1, 1], [0, 1], [0, 1]]  # Left
        ]
    }
    
    @staticmethod
    def get_piece_shape(piece_type, rotation):
        """Get the shape matrix for a piece at given rotation"""
        if piece_type not in TetrisPiece.PIECES:
            return [[1]]  # Default fallback
        
        rotations = TetrisPiece.PIECES[piece_type]
        rotation_index = rotation % len(rotations)
        return rotations[rotation_index]

class EnhancedTetrisEnvironment:
    """Enhanced environment with proper game simulation"""
    
    def __init__(self):
        self.piece_types = ['I', 'O', 'T', 'J', 'L', 'S', 'Z']
    
    def decode_action(self, action, board_width=10):
        """
        Decode action into position and rotation
        Assuming action encoding: action = position + rotation * board_width
        """
        position = action % board_width
        rotation = action // board_width
        return position, rotation
    
    def can_place_piece(self, board, piece_shape, position, board_height, board_width):
        """Check if a piece can be placed at the given position"""
        piece_height = len(piece_shape)
        piece_width = len(piece_shape[0]) if piece_shape else 0
        
        # Check bounds
        if position < 0 or position + piece_width > board_width:
            return False, -1
        
        # Convert 1D board to 2D
        board_2d = np.array(board).reshape(board_height, board_width)
        
        # Find the lowest valid row for this piece
        for row in range(board_height - piece_height + 1):
            valid = True
            for p_row in range(piece_height):
                for p_col in range(piece_width):
                    if piece_shape[p_row][p_col] == 1:
                        board_row = row + p_row
                        board_col = position + p_col
                        if board_2d[board_row, board_col] != 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Check if piece would "fall" further
                can_fall = False
                if row + piece_height < board_height:
                    for p_row in range(piece_height):
                        for p_col in range(piece_width):
                            if piece_shape[p_row][p_col] == 1:
                                board_row = row + piece_height
                                board_col = position + p_col
                                if board_row < board_height and board_2d[board_row, board_col] == 0:
                                    can_fall = True
                                    break
                        if can_fall:
                            break
                
                if not can_fall:
                    return True, row
        
        return False, -1
    
    def place_piece(self, board, piece_shape, position, row, board_height, board_width):
        """Place a piece on the board and return the new board"""
        board_2d = np.array(board).reshape(board_height, board_width)
        new_board = board_2d.copy()
        
        piece_height = len(piece_shape)
        piece_width = len(piece_shape[0]) if piece_shape else 0
        
        # Place the piece
        for p_row in range(piece_height):
            for p_col in range(piece_width):
                if piece_shape[p_row][p_col] == 1:
                    board_row = row + p_row
                    board_col = position + p_col
                    if 0 <= board_row < board_height and 0 <= board_col < board_width:
                        new_board[board_row, board_col] = 1
        
        return new_board.flatten().tolist()
    
    def clear_lines(self, board, board_height, board_width):
        """Clear completed lines and return new board and lines cleared"""
        board_2d = np.array(board).reshape(board_height, board_width)
        lines_cleared = 0
        
        # Check each row from bottom to top
        new_board = []
        for row in range(board_height - 1, -1, -1):
            if np.all(board_2d[row] != 0):  # Line is complete
                lines_cleared += 1
            else:
                new_board.append(board_2d[row].tolist())
        
        # Add empty rows at the top
        while len(new_board) < board_height:
            new_board.append([0] * board_width)
        
        # Reverse to get correct order (top to bottom)
        new_board.reverse()
        
        return np.array(new_board).flatten().tolist(), lines_cleared
    
    def calculate_column_heights(self, board, board_height, board_width):
        """Calculate height of each column"""
        board_2d = np.array(board).reshape(board_height, board_width)
        heights = []
        
        for col in range(board_width):
            col_height = 0
            for row in range(board_height):
                if board_2d[row, col] != 0:
                    col_height = board_height - row
                    break
            heights.append(col_height)
        
        return heights
    
    def calculate_holes(self, board, board_height, board_width):
        """Calculate number of holes in the board"""
        holes = 0
        board_2d = np.array(board).reshape(board_height, board_width)
        
        for col in range(board_width):
            # Find the topmost filled cell in this column
            top_filled = -1
            for row in range(board_height):
                if board_2d[row, col] != 0:
                    top_filled = row
                    break
            
            # Count holes below the topmost filled cell
            if top_filled != -1:
                for row in range(top_filled + 1, board_height):
                    if board_2d[row, col] == 0:
                        holes += 1
        
        return holes
    
    def calculate_bumpiness(self, heights):
        """Calculate bumpiness (sum of height differences between adjacent columns)"""
        if len(heights) < 2:
            return 0
        
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return bumpiness
    
    def calculate_additional_features(self, board, heights, board_height, board_width):
        """Calculate additional features for better evaluation"""
        board_2d = np.array(board).reshape(board_height, board_width)
        
        # Wells (deep holes)
        wells = 0
        for col in range(board_width):
            left_height = heights[col - 1] if col > 0 else 0
            right_height = heights[col + 1] if col < board_width - 1 else 0
            current_height = heights[col]
            
            if current_height < left_height and current_height < right_height:
                wells += min(left_height, right_height) - current_height
        
        # Row transitions (horizontal discontinuities)
        row_transitions = 0
        for row in range(board_height):
            for col in range(board_width - 1):
                if (board_2d[row, col] == 0) != (board_2d[row, col + 1] == 0):
                    row_transitions += 1
        
        # Column transitions (vertical discontinuities)
        col_transitions = 0
        for col in range(board_width):
            for row in range(board_height - 1):
                if (board_2d[row, col] == 0) != (board_2d[row + 1, col] == 0):
                    col_transitions += 1
        
        return wells, row_transitions, col_transitions
    
    def extract_features(self, game_state):
        """Extract enhanced features from game state"""
        board_height = int(game_state.get('curriculumBoardHeight', 20))
        board_width = 10
        
        board = game_state.get('board', [])
        if not board:
            board = [0] * (board_height * board_width)
        
        # Ensure board is the right size
        expected_size = board_height * board_width
        if len(board) != expected_size:
            if len(board) < expected_size:
                board = board + [0] * (expected_size - len(board))
            else:
                board = board[:expected_size]
        
        # Calculate or get heights
        if 'heights' in game_state and game_state['heights']:
            heights = game_state['heights'][:board_width]
            while len(heights) < board_width:
                heights.append(0)
        else:
            heights = self.calculate_column_heights(board, board_height, board_width)
        
        # Calculate basic features
        lines_cleared = game_state.get('linesCleared', 0)
        holes = game_state.get('holesCount', self.calculate_holes(board, board_height, board_width))
        bumpiness = game_state.get('bumpiness', self.calculate_bumpiness(heights))
        total_height = sum(heights)
        
        # Calculate additional features
        wells, row_transitions, col_transitions = self.calculate_additional_features(
            board, heights, board_height, board_width
        )
        
        # Max height
        max_height = max(heights) if heights else 0
        
        # Normalize features
        normalized_features = [
            lines_cleared / 4.0,                           # Lines cleared
            holes / 20.0,                                  # Holes
            bumpiness / 50.0,                              # Bumpiness
            total_height / (board_height * board_width),   # Total height
            wells / 10.0,                                  # Wells
            row_transitions / (board_height * board_width), # Row transitions
            col_transitions / (board_height * board_width), # Column transitions
            max_height / board_height                       # Max height
        ]
        
        return np.array(normalized_features, dtype=np.float32)

class SimpleTetrisDQN(nn.Module):
    """Enhanced DQN with more input features"""
    def __init__(self, input_size=8, hidden_size=64, output_size=1):
        super(SimpleTetrisDQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class EnhancedDQNAgent:
    """Enhanced DQN Agent with proper action simulation"""
    
    def __init__(self, lr=0.001, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 tensorboard_log_dir=None):
        self.device = device
        self.memory = deque(maxlen=20000)
        self.learning_rate = lr
        self.batch_size = 512
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_decay_episodes = 1500
        self.steps = 0
        self.episodes = 0
        self.target_update_freq = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_episodes = 0
        self.piece_types = ['I', 'O', 'T', 'J', 'L', 'S', 'Z']

        
        # TensorBoard writer
        if tensorboard_log_dir is None:
            tensorboard_log_dir = f"runs/enhanced_tetris_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(tensorboard_log_dir)
        
        # Enhanced neural network with 8 features
        self.q_network = SimpleTetrisDQN(input_size=8, hidden_size=64, output_size=1).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Enhanced environment
        self.env = EnhancedTetrisEnvironment()
        
        print(f"Enhanced DQN Agent initialized on {device}")
        print(f"TensorBoard logs: {tensorboard_log_dir}")
    
    def simulate_action(self, state, action):
        """
        Enhanced action simulation with proper piece placement
        """
        board_height = int(state.get('curriculumBoardHeight', 20))
        board_width = 10
        
        # Get current board
        board = state.get('board', [0] * (board_height * board_width))[:]
        
        # Get current piece info (Unity now sends [typeIndex, rotation, x, y])
        piece_info = state.get('currentPiece', [2, 0, 0, 0])  # default to T (index 2)
        piece_index = int(piece_info[0])
        # self.env.piece_types == ['I','O','T','S','Z','J','L']
        if 0 <= piece_index < len(self.env.piece_types):
            current_piece = self.env.piece_types[piece_index]
        else:
            current_piece = 'T'

        
        # Decode action into position and rotation
        position, rotation = self.env.decode_action(action, board_width)
        print(current_piece,position,rotation)
        # Get piece shape
        piece_shape = TetrisPiece.get_piece_shape(current_piece, rotation)
        
        # Check if piece can be placed
        can_place, drop_row = self.env.can_place_piece(
            board, piece_shape, position, board_height, board_width
        )
        
        if not can_place:
            # Invalid action - return current state with penalty
            simulated_state = state.copy()
            simulated_state['linesCleared'] = 0
            simulated_state['invalid_action'] = True
            return simulated_state
        
        # Place the piece
        new_board = self.env.place_piece(
            board, piece_shape, position, drop_row, board_height, board_width
        )
        
        # Clear completed lines
        final_board, lines_cleared = self.env.clear_lines(
            new_board, board_height, board_width
        )
        
        # Calculate new heights
        new_heights = self.env.calculate_column_heights(
            final_board, board_height, board_width
        )
        
        # Calculate new holes and bumpiness
        new_holes = self.env.calculate_holes(final_board, board_height, board_width)
        new_bumpiness = self.env.calculate_bumpiness(new_heights)
        
        # Create simulated state
        simulated_state = {
            'board': final_board,
            'curriculumBoardHeight': board_height,
            'heights': new_heights,
            'linesCleared': lines_cleared,
            'holesCount': new_holes,
            'bumpiness': new_bumpiness,
            'validActions': state.get('validActions', []),
            'currentPiece': current_piece,
            'invalid_action': False
        }
        
        return simulated_state
    
    def get_possible_states(self, current_state):
        """Generate all possible next states for valid actions"""
        valid_actions = current_state.get('validActions', [])
        possible_states = []
        
        for action in valid_actions:
            # Simulate the action
            simulated_state = self.simulate_action(current_state, action)
            
            # Skip invalid actions
            if simulated_state.get('invalid_action', False):
                continue
            
            # Extract features from the simulated state
            features = self.env.extract_features(simulated_state)
            possible_states.append((action, features, simulated_state))
        
        return possible_states
    
    def calculate_reward(self, old_state, new_state):
        """Calculate reward based on state transition"""
        reward = 0
        
        # Reward for clearing lines
        lines_cleared = new_state.get('linesCleared', 0)
        if lines_cleared > 0:
            # Exponential reward for more lines cleared at once
            line_rewards = {1: 40, 2: 100, 3: 300, 4: 1200}
            reward += line_rewards.get(lines_cleared, lines_cleared * 40)
        
        # Get heights
        old_heights = old_state.get('heights', [0] * 10)
        new_heights = new_state.get('heights', [0] * 10)
        
        # Penalty for increasing height
        height_diff = sum(new_heights) - sum(old_heights)
        reward -= height_diff * 0.5
        
        # Penalty for creating holes
        old_holes = old_state.get('holesCount', 0)
        new_holes = new_state.get('holesCount', 0)
        holes_created = new_holes - old_holes
        reward -= holes_created * 2
        
        # Penalty for increasing bumpiness
        old_bumpiness = old_state.get('bumpiness', 0)
        new_bumpiness = new_state.get('bumpiness', 0)
        bumpiness_diff = new_bumpiness - old_bumpiness
        reward -= bumpiness_diff * 0.1
        
        # Penalty for invalid action
        if new_state.get('invalid_action', False):
            reward -= 10
        
        return reward
    
    def get_current_epsilon(self):
        """Calculate current epsilon based on episode"""
        if self.episodes >= self.epsilon_decay_episodes:
            return self.epsilon_min
        else:
            decay_ratio = self.episodes / self.epsilon_decay_episodes
            return self.epsilon * (1 - decay_ratio) + self.epsilon_min * decay_ratio
    
    def act(self, state, training=True):
        """Choose action using enhanced evaluation"""
        current_epsilon = self.get_current_epsilon()
        
        # Epsilon-greedy exploration
        if training and random.random() < current_epsilon:
            valid = state.get('validActions', [])
            if training and valid and random.random() < current_epsilon:
                action = random.choice(valid)
            elif training and not valid:
                # no valid moves, just return 0 or some safe default
                action = 0
            self.writer.add_scalar('Agent/Random_Action_Taken', 1, self.steps)
            self.writer.add_scalar('Agent/Epsilon', current_epsilon, self.steps)
            return action
        
        # Get all possible states
        possible_states = self.get_possible_states(state)
        
        if not possible_states:
            return state.get('validActions', [0])[0] if state.get('validActions') else 0
        
        # Evaluate each possible state
        best_action = possible_states[0][0]
        best_q_value = float('-inf')
        
        with torch.no_grad():
            for action, features, simulated_state in possible_states:
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                q_value = self.q_network(features_tensor).item()
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
        
        # Log metrics
        if training:
            self.writer.add_scalar('Agent/Random_Action_Taken', 0, self.steps)
            self.writer.add_scalar('Agent/Epsilon', current_epsilon, self.steps)
            self.writer.add_scalar('Agent/Best_Q_Value', best_q_value, self.steps)
            self.writer.add_scalar('Agent/Possible_Actions', len(possible_states), self.steps)
        
        self.steps += 1
        return best_action
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        state_features = self.env.extract_features(state)
        next_state_features = self.env.extract_features(next_state) if next_state else None
        
        self.memory.append((state_features, action, reward, next_state_features, done))
        
        if len(self.memory) % 1000 == 0:
            self.writer.add_scalar('Agent/Memory_Size', len(self.memory), self.steps)
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = []
        targets = []
        
        for state_features, action, reward, next_state_features, done in batch:
            target = reward
            
            if not done and next_state_features is not None:
                next_state_tensor = torch.FloatTensor(next_state_features).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    next_q_value = self.q_network(next_state_tensor).item()
                target = reward + self.gamma * next_q_value
            
            states.append(state_features)
            targets.append(target)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        targets_tensor = torch.FloatTensor(targets).unsqueeze(1).to(self.device)
        
        # Forward pass
        current_q_values = self.q_network(states_tensor)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, targets_tensor)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Log metrics
        self.writer.add_scalar('Training/Loss', loss.item(), self.steps)
        
        return loss.item()
    
    def end_episode(self, episode_reward, episode_length, episode_score):
        """Called at the end of each episode"""
        self.episodes += 1
        
        # Log episode metrics
        self.writer.add_scalar('Episode/Reward', episode_reward, self.episodes)
        self.writer.add_scalar('Episode/Length', episode_length, self.episodes)
        self.writer.add_scalar('Episode/Score', episode_score, self.episodes)
        self.writer.add_scalar('Episode/Epsilon', self.get_current_epsilon(), self.episodes)
        
        print(f"Episode {self.episodes}: Score={episode_score}, Reward={episode_reward:.2f}, "
              f"Length={episode_length}, Epsilon={self.get_current_epsilon():.3f}")
    
    def save(self, filepath):
        """Save the model"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episodes': self.episodes,
            'steps': self.steps,
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episodes = checkpoint.get('episodes', 0)
            self.steps = checkpoint.get('steps', 0)
            print(f"Model loaded from {filepath}")
            return True
        return False
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()
    def log_episode_metrics(self, episode, episode_reward, episode_length, episode_score, 
                          episode_lines, game_metrics):
        """Log episode metrics to TensorBoard"""
        # FIX 4: Increment episode counter
        self.training_episodes += 1
        
        self.writer.add_scalar('Episode/Reward', episode_reward, episode)
        self.writer.add_scalar('Episode/Length', episode_length, episode)
        self.writer.add_scalar('Episode/Score', episode_score, episode)
        self.writer.add_scalar('Episode/Lines_Cleared', episode_lines, episode)
        
        # Game-specific metrics
        self.writer.add_scalar('Game/Holes_Count', game_metrics.get('holes_count', 0), episode)
        self.writer.add_scalar('Game/Stack_Height', game_metrics.get('stack_height', 0), episode)
        self.writer.add_scalar('Game/Perfect_Clear', 1 if game_metrics.get('perfect_clear', False) else 0, episode)
        
        # Running averages
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        if len(self.episode_rewards) >= 100:
            avg_reward_100 = np.mean(self.episode_rewards[-100:])
            avg_length_100 = np.mean(self.episode_lengths[-100:])
            self.writer.add_scalar('Episode/Avg_Reward_100', avg_reward_100, episode)
            self.writer.add_scalar('Episode/Avg_Length_100', avg_length_100, episode)
        
        if len(self.episode_rewards) >= 10:
            avg_reward_10 = np.mean(self.episode_rewards[-10:])
            avg_length_10 = np.mean(self.episode_lengths[-10:])
            self.writer.add_scalar('Episode/Avg_Reward_10', avg_reward_10, episode)
            self.writer.add_scalar('Episode/Avg_Length_10', avg_length_10, episode)        

# Example usage
if __name__ == "__main__":
    # Create enhanced agent
    agent = EnhancedDQNAgent()
    
    # Example game state with more realistic data
    example_state = {
        'board': [0] * 200,  # 20x10 board
        'curriculumBoardHeight': 20,
        'heights': [0, 1, 2, 0, 0, 3, 1, 0, 0, 0],
        'validActions': [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24],  # pos + rot*10
        'linesCleared': 0,
        'holesCount': 0,
        'bumpiness': 3,
        'currentPiece': 'T'
    }
    
    # Test action simulation
    action = 12  # Position 2, Rotation 1
    simulated_state = agent.simulate_action(example_state, action)
    print(f"Original heights: {example_state['heights']}")
    print(f"Simulated heights: {simulated_state['heights']}")
    print(f"Lines cleared: {simulated_state['linesCleared']}")
    
    # Choose action
    best_action = agent.act(example_state)
    print(f"Chosen action: {best_action}")
    
    agent.close()

