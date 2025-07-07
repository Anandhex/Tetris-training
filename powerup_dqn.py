import numpy as np
import random
import logging
from collections import deque
import onnxruntime as ort
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tetris_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TetrisAI')

class Tetris:
    SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1], [1, 1]],  # O
        [[0, 1, 0], [1, 1, 1]],  # T
        [[1, 1, 0], [0, 1, 1]],  # S
        [[0, 1, 1], [1, 1, 0]],  # Z
        [[1, 0, 0], [1, 1, 1]],  # J
        [[0, 0, 1], [1, 1, 1]]   # L
    ]
    
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.reset()
        logger.debug(f"Initialized Tetris board {width}x{height}")
        
    def reset(self):
        self.board = np.zeros((self.height, self.width), dtype=int)
        self.block_count = 0
        self.current_powerup = None
        self.last_placement = None
        self.game_over = False
        self.score = 0
        self._new_block()
        logger.debug("Game reset")
        return self.get_state()
    
    def _new_block(self):
        self.current_block = random.choice(self.SHAPES)
        self.block_pos = [0, self.width // 2 - len(self.current_block[0]) // 2]
        logger.debug(f"New block generated: {self.current_block}")
        
    def get_state(self):
        powerup_vec = np.zeros(3)
        if self.current_powerup == 'bottom_clear':
            powerup_vec[0] = 1
        elif self.current_powerup == 'gravity':
            powerup_vec[1] = 1
        elif self.current_powerup == 'bomb':
            powerup_vec[2] = 1
        return self.board.copy(), powerup_vec
    
    def place_block(self, placement_model, model_type='keras'):
        if self.game_over:
            logger.debug("Game over. Cannot place block.")
            return 0, True
            
        best_score = -float('inf')
        best_action = None
        
        # Store features and corresponding actions to pass to the model
        all_features = []
        possible_actions = [] # (block, original_block_pos, rot, x)
        
        logger.debug(f"Evaluating placements for block: {self.current_block}")
        
        for rot in range(4):
            block = np.rot90(self.current_block, k=rot)
            block_width = len(block[0])
            
            # Simulate hard drop to find the final y position for each x
            for x in range(self.width - block_width + 1):
                test_pos_initial = [0, x] # Start at top
                
                # Simulate dropping the block
                sim_board_temp = self.board.copy() # Board for collision check during drop
                current_drop_y = 0
                
                # Find the lowest valid y position
                while not self._check_collision(block, [current_drop_y + 1, x], sim_board_temp) and current_drop_y + 1 + len(block) <= self.height:
                    current_drop_y += 1
                
                final_drop_pos = [current_drop_y, x]
                
                # Check if this placement is valid on the final dropped position
                if not self._check_collision(block, final_drop_pos):
                    test_board = self._simulate_placement(block, final_drop_pos)
                    
                    # Calculate features for this simulated board
                    lines_cleared = self._calculate_lines_cleared_sim(test_board)
                    holes = self._count_holes_sim(test_board)
                    bumpiness = self._get_bumpiness_score_sim(test_board)
                    height = self._calculate_stack_height_sim(test_board)
                    
                    features = [lines_cleared, holes, bumpiness, height]
                    all_features.append(features)
                    possible_actions.append((block, final_drop_pos)) # Store block and final position
                else:
                    # If even the initial placement or dropped placement is invalid, it's not a possible move
                    logger.debug(f"Skipping invalid placement: rot={rot}, x={x}")


        if not all_features:
            logger.warning("No valid placement found after simulation! Game over.")
            self.game_over = True
            return 0, True

        features_array = np.array(all_features, dtype=np.float32)

        if model_type == 'onnx':
            ort_inputs = {placement_model.get_inputs()[0].name: features_array}
            ort_outs = placement_model.run(None, ort_inputs)
            scores = ort_outs[0].flatten() # Output is likely (num_placements, 1) or (num_placements,)
        else: # Keras
            scores = placement_model.predict(features_array, verbose=0).flatten() # Ensure it's 1D

        best_index = np.argmax(scores)
        best_score = scores[best_index]
        block, pos = possible_actions[best_index]

        logger.debug(f"Placing best block at position: y={pos[0]}, x={pos[1]}, score={best_score:.4f}")
        self._place(block, pos)
        
        # Clear lines after actual placement
        cleared = self._clear_lines()
        self.score += cleared * 100
        self.block_count += 1
        self.last_placement = (pos[0] + len(block) - 1, pos[1] + len(block[0]) // 2)
        logger.debug(f"Placed block. Cleared {cleared} lines. Total score: {self.score}")
        
        # Assign new powerup every 5 blocks
        if self.block_count % 5 == 0 and self.block_count > 0:
            self.current_powerup = random.choice(['bottom_clear', 'gravity', 'bomb'])
            logger.info(f"New powerup assigned: {self.current_powerup}")
            
        return cleared * 100, self.game_over
    
    def use_powerup(self):
        if not self.current_powerup or self.game_over:
            logger.debug("No powerup to use or game over")
            return 0, self.board.copy()
            
        original_board = self.board.copy()
        reward = 0
        powerup_type = self.current_powerup
        logger.info(f"Using powerup: {powerup_type}")
        
        if powerup_type == 'bottom_clear':
            reward = self._clear_bottom_line()
            
        elif powerup_type == 'gravity':
            reward = self._apply_gravity()
            
        elif powerup_type == 'bomb':
            bomb_position = self._find_best_bomb_position()
            logger.debug(f"Bomb placed at: {bomb_position}")
            reward = self._use_bomb(bomb_position)
        
        # Clear any lines created by powerup
        cleared = self._clear_lines()
        reward += cleared * 100
        self.score += reward
        
        logger.info(f"Powerup result: +{reward} points (cleared {cleared} lines)")
        self.current_powerup = None
        return reward, original_board
    
    def _find_best_bomb_position(self):
        """Find position that maximizes blocks destroyed in 3x3 area"""
        best_pos = (self.height // 2, self.width // 2)
        max_destroyed = 0
        
        logger.debug("Finding optimal bomb position")
        for y in range(1, self.height-1):
            for x in range(1, self.width-1):
                destroyed = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.height and 0 <= nx < self.width and self.board[ny, nx]:
                            destroyed += 1
                if destroyed > max_destroyed:
                    max_destroyed = destroyed
                    best_pos = (y, x)
        
        logger.debug(f"Best bomb position: {best_pos} (destroys {max_destroyed} blocks)")
        return best_pos
    
    def _clear_bottom_line(self):
        """Clear bottom line powerup with reward calculation"""
        blocks_cleared = np.sum(self.board[-1])
        if blocks_cleared == 0:
            logger.debug("Bottom line clear: Nothing to clear")
            return 0
        
        logger.debug(f"Clearing bottom line: {blocks_cleared} blocks")
        self.board[-1] = 0
        for i in range(self.height-2, -1, -1):
            self.board[i+1] = self.board[i]
        self.board[0] = 0
        return blocks_cleared * 5
    
    def _count_holes(self):
        """Count holes (empty spaces with block above) on the actual board"""
        holes = 0
        for col in range(self.width):
            found_block = False
            for row in range(self.height-1, -1, -1):
                if self.board[row, col]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes
    
    def _apply_gravity(self):
        """Apply gravity powerup - pull down one line with reward"""
        original_holes = self._count_holes()
        logger.debug(f"Applying gravity. Original holes: {original_holes}")
        
        new_board = np.zeros_like(self.board)
        for col in range(self.width):
            col_blocks = []
            for row in range(self.height):
                if self.board[row, col]:
                    col_blocks.append(1)
            if col_blocks:
                start_row = self.height - len(col_blocks)
                new_board[start_row:, col] = 1
        
        self.board = new_board
        new_holes = self._count_holes()
        holes_filled = original_holes - new_holes
        
        logger.debug(f"Gravity result: Filled {holes_filled} holes")
        return holes_filled * 3
    
    def _use_bomb(self, position):
        """Use bomb powerup - destroy 3x3 area with reward"""
        y, x = position
        destroyed = 0
        logger.debug(f"Exploding bomb at ({y}, {x})")
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if self.board[ny, nx]:
                        destroyed += 1
                    self.board[ny, nx] = 0
        
        logger.debug(f"Bomb destroyed {destroyed} blocks")
        return destroyed * 8
    
    # Modified _check_collision to accept an optional board argument
    def _check_collision(self, block, pos, board_to_check=None):
        if board_to_check is None:
            board_to_check = self.board # Default to current game board
        y, x = pos
        for i in range(len(block)):
            for j in range(len(block[0])):
                if block[i][j]:
                    if (y + i >= self.height or # Collided with bottom
                        x + j < 0 or # Collided with left wall
                        x + j >= self.width or # Collided with right wall
                        (y + i >= 0 and board_to_check[y + i, x + j])): # Collided with existing block
                        return True
        return False
    
    def _place(self, block, pos):
        y, x = pos
        for i in range(len(block)):
            for j in range(len(block[0])):
                if block[i][j]:
                    # Check if placement goes above the board, indicating game over
                    if y + i < 0:
                        logger.warning("Block placed above board! Game over.")
                        self.game_over = True
                        return
                    if y + i < self.height: # Ensure it's within bounds vertically
                        self.board[y + i, x + j] = 1
        self._new_block()
    
    def _clear_lines(self):
        full_lines = []
        for i in range(self.height):
            if all(self.board[i]):
                full_lines.append(i)
        
        for line in full_lines:
            self.board = np.delete(self.board, line, axis=0)
            self.board = np.vstack([np.zeros((1, self.width)), self.board])
        
        if full_lines:
            logger.debug(f"Cleared {len(full_lines)} lines")
        return len(full_lines)
    
    def _simulate_placement(self, block, pos):
        sim_board = self.board.copy()
        y, x = pos
        for i in range(len(block)):
            for j in range(len(block[0])):
                if block[i][j]:
                    # Assuming collision check has already ensured validity for the final position
                    sim_board[y + i, x + j] = 1
        return sim_board

    # --- Feature Calculation for Simulated Boards (Python equivalents of C# logic) ---

    def _calculate_lines_cleared_sim(self, sim_board):
        """Calculates how many lines would be cleared on a simulated board."""
        full_lines = 0
        for r in range(self.height):
            if all(sim_board[r]):
                full_lines += 1
        return full_lines

    def _count_holes_sim(self, sim_board):
        """Counts holes on a simulated board."""
        holes = 0
        for col in range(self.width):
            found_block = False
            for row in range(self.height): # Iterate from top to bottom
                if sim_board[row, col] == 1:
                    found_block = True
                elif found_block and sim_board[row, col] == 0:
                    holes += 1
        return holes

    def _get_bumpiness_score_sim(self, sim_board):
        """Calculates bumpiness score for a simulated board."""
        heights = np.zeros(self.width, dtype=int)
        for col in range(self.width):
            for row in range(self.height):
                if sim_board[row, col] == 1:
                    heights[col] = self.height - row # Height from top
                    break
        
        bumpiness = 0
        for i in range(self.width - 1):
            bumpiness += abs(heights[i] - heights[i+1])
        return bumpiness

    def _calculate_stack_height_sim(self, sim_board):
        """Calculates maximum height of the stack on a simulated board."""
        for row in range(self.height):
            if np.any(sim_board[row] == 1):
                return self.height - row
        return 0 # Empty board

class PowerupDQNAgent:
    def __init__(self, board_size, powerup_size):
        self.board_size = board_size
        self.powerup_size = powerup_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        logger.info("Powerup DQN Agent initialized")
    
    def _build_model(self):
        # Input to PowerupDQNAgent model is board_size (flattened board) + powerup_size
        model = Sequential([
            Dense(256, input_dim=self.board_size + self.powerup_size, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(2, activation='linear') # Output for 2 actions: Use powerup (1) or Don't use (0)
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Updated target model weights")
    
    def remember(self, board, powerup, action, reward, next_board, next_powerup, done):
        self.memory.append((board, powerup, action, reward, next_board, next_powerup, done))
    
    def act(self, board, powerup):
        if np.random.rand() <= self.epsilon:
            action = random.choice([0, 1])
            logger.debug(f"Random action: {action}")
            return action
        
        flat_board = board.flatten()
        state = np.concatenate([flat_board, powerup]).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)
        action = np.argmax(q_values[0])
        logger.debug(f"Model action: {action}, Q-values: {q_values[0]}")
        return action
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            logger.debug("Not enough samples for replay")
            return
            
        minibatch = random.sample(self.memory, batch_size)
        logger.debug(f"Running replay with {batch_size} samples")
        
        for board, powerup, action, reward, next_board, next_powerup, done in minibatch:
            flat_board = board.flatten()
            state = np.concatenate([flat_board, powerup])
            next_flat_board = next_board.flatten()
            next_state = np.concatenate([next_flat_board, next_powerup])
            
            target = self.model.predict(state.reshape(1, -1), verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state.reshape(1, -1), verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t)
            
            self.model.fit(state.reshape(1, -1), target, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            logger.debug(f"Epsilon decayed to: {self.epsilon:.4f}")
    
    def save(self, name):
        self.model.save(name)
        logger.info(f"Saved model as {name}")

def load_block_placement_model(model_path, model_type='keras'):
    logger.info(f"Loading block placement model: {model_path}")
    if model_type == 'onnx':
        return ort.InferenceSession(model_path)
    else:
        # For Keras, if you load a model that expects (None, 4) features, it will work.
        # This assumes your 'tetris_model.keras' is the one that expects 4 features.
        return load_model(model_path)

# Training parameters
EPISODES = 1000
BATCH_SIZE = 32
UPDATE_TARGET_EVERY = 50

def main():
    logger.info("Starting Tetris AI training")
    
    # Initialize game environment
    env = Tetris(width=10, height=20)
    board_size = env.height * env.width # This is for the PowerupDQN agent input
    
    # Load block placement model
    try:
        # Assuming tetris_model.keras is the one expecting 4 features now
        block_placement_model = load_block_placement_model('model/tetris_model.keras', 'keras')
        model_type = 'keras'
        logger.info("Using Keras model for block placement (4 features input)")
    except Exception as e:
        logger.warning(f"Keras model not found or incompatible: {e}, trying ONNX model")
        try:
            block_placement_model = load_block_placement_model('model/tetris_dqn.onnx', 'onnx') # This is the ONNX model expecting 4 features
            model_type = 'onnx'
            logger.info("Using ONNX model for block placement (4 features input)")
        except Exception as e:
            logger.error(f"Failed to load any model: {e}")
            return
    
    # Initialize powerup agent
    powerup_agent = PowerupDQNAgent(board_size=board_size, powerup_size=3)
    
    # Training loop
    for episode in range(EPISODES):
        board, powerup = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        logger.info(f"Starting episode {episode+1}/{EPISODES}")
        
        while not done:
            # Place block using pre-trained model (now expects features)
            block_reward, done = env.place_block(block_placement_model, model_type)
            total_reward += block_reward
            
            if done:
                logger.info(f"Episode {episode+1} ended early at step {step}")
                break
            
            # Power-up decision
            if env.current_powerup:
                current_board, current_powerup_vec = env.get_state()
                action = powerup_agent.act(current_board, current_powerup_vec)
                
                if action == 1:  # Use power-up
                    powerup_reward, original_board_state_before_powerup = env.use_powerup()
                    total_reward += powerup_reward
                    next_board, next_powerup_vec = env.get_state()
                    done = env.game_over
                    
                    # Store experience
                    powerup_agent.remember(
                        original_board_state_before_powerup, current_powerup_vec, # Use original board state before powerup as state
                        action, powerup_reward, 
                        next_board, next_powerup_vec, 
                        done
                    )
                    logger.debug("Stored powerup usage experience")
                else:
                    # If not using power-up, store with no immediate change, reward is 0
                    # The state and next state should ideally reflect that the powerup was NOT used
                    next_board_state_after_block_placement, next_powerup_vec = env.get_state() # Get state AFTER block placement
                    powerup_agent.remember(
                        current_board, current_powerup_vec, # The state when the decision was made
                        action, 0, # Reward for not using powerup
                        next_board_state_after_block_placement, next_powerup_vec, # Next state is after block placement (no powerup used)
                        False # Not done just because we didn't use a powerup
                    )
                    logger.debug("Stored powerup skip experience")
                
                # Train agent periodically
                if step % 5 == 0:
                    powerup_agent.replay(BATCH_SIZE)
                
                # Update target network periodically
                if step % UPDATE_TARGET_EVERY == 0:
                    powerup_agent.update_target_model()
                
            step += 1
        
        logger.info(f"Episode {episode+1} completed. Total Reward: {total_reward}, Epsilon: {powerup_agent.epsilon:.4f}")
    
    # Save trained model
    powerup_agent.save("model/powerup_dqn.keras")
    logger.info("Training complete. Saved powerup DQN model")

if __name__ == "__main__":
    main()