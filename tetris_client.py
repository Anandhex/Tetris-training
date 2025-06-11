# tetris_client.py
import socket
import json
import numpy as np
import time
import threading
from queue import Queue
import random
from collections import deque
import pickle
import os
#12347 without noise
class UnityTetrisClient:
    def __init__(self, host='127.0.0.1', port=12348):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.game_state_queue = Queue()
        self.running = False
        self.action_space_size = 40
        self.board_width = 10
        self.num_rotations = 4
    
    def get_curriculum_info(self, game_state):
        """Extract curriculum information from game state"""
        return {
            'board_height': game_state.get('curriculumBoardHeight', 20),
            'board_preset': game_state.get('curriculumBoardPreset', 0),
            'allowed_tetromino_types': game_state.get('allowedTetrominoTypes', 7)
        }
    
    def is_game_over(self, game_state):
        """Check if the game is over"""
        return game_state.get('gameOver', False) or game_state.get('episodeEnd', False)
    
    def get_action_space_info(self, game_state):
        """Get action space information from game state"""
        return {
            'action_space_size': game_state.get('actionSpaceSize', 40),
            'action_space_type': game_state.get('actionSpaceType', 'column_rotation'),
            'is_executing_action': game_state.get('isExecutingAction', False),
            'waiting_for_action': game_state.get('waitingForAction', False)
        }
    
    def get_board_metrics(self, game_state):
        """Get board analysis metrics"""
        return {
            'holes_count': game_state.get('holesCount', 0),
            'stack_height': game_state.get('stackHeight', 0),
            'perfect_clear': game_state.get('perfectClear', False),
            'lines_cleared': game_state.get('linesCleared', 0)
        }
    
    def wait_for_game_ready(self, timeout=10.0, check_interval=0.1):
        """Wait until the game is ready to receive actions"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            state = self.get_game_state(timeout=check_interval)
            if state:
                action_info = self.get_action_space_info(state)
                if action_info['waiting_for_action'] and not action_info['is_executing_action']:
                    return state
            time.sleep(check_interval)
        
        return None
    
    def send_action_and_wait(self, action_index, timeout=5.0):
        """Send action and wait for the result state"""
        if not self.send_action(action_index):
            return None
        
        # Wait for the action to complete and get the resulting state
        start_time = time.time()
        while time.time() - start_time < timeout:
            state = self.get_game_state(timeout=0.1)
            if state:
                action_info = self.get_action_space_info(state)
                # Return state when action is complete
                if not action_info['is_executing_action']:
                    return state
        
        return None
    def connect(self, max_retries=5, retry_delay=2.0):
        """Connect to Unity with retries"""
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to Unity... (attempt {attempt + 1}/{max_retries})")
                
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10.0)
                self.socket.connect((self.host, self.port))
                
                self.connected = True
                self.running = True
                
                self.receive_thread = threading.Thread(target=self._receive_loop)
                self.receive_thread.daemon = True
                self.receive_thread.start()
                
                print(f" Connected to Unity at {self.host}:{self.port}")
                return True
                
            except Exception as e:
                print(f" Connection failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                
        return False
    
    def _receive_loop(self):
        buffer = ""
        while self.running and self.connected:
            try:
                data = self.socket.recv(4096).decode('utf-8')
                if not data:
                    break
                    
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            game_state = json.loads(line)
                            self.game_state_queue.put(game_state)
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                break
                
        self.connected = False
    
    def send_action(self, action_index):
        if not self.connected:
            return False
            
        if action_index < 0 or action_index >= self.action_space_size:
            return False
        
        command = {
            "type": "action",
            "action": {"actionIndex": int(action_index)}
        }
        
        return self._send_command(command)
    
    def send_curriculum_change(self, board_height=20, board_preset=0, tetromino_types=7,stage_name=None):
        command = {
            "type": "curriculum_change",
            "curriculum": {
                "boardHeight": board_height,
                "boardPreset": board_preset,
                "allowedTetrominoTypes": tetromino_types,
                "stageName": stage_name or "Unknown",
                "timestamp": time.time()
            }
        }
        return self._send_command(command)
    def send_curriculum_status_request(self):
        """Request current curriculum status from Unity"""
        command = {
            "type": "curriculum_status_request"
        }
        return self._send_command(command)
    
    def get_curriculum_confirmation(self, timeout=2.0):
        """Wait for curriculum change confirmation"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            state = self.get_game_state(timeout=0.1)
            if state and 'curriculumConfirmed' in state:
                return state
        return None
    def send_reset(self):
        command = {
            "type": "reset",
            "reset": {"resetBoard": True}
        }
        return self._send_command(command)
    
    def _send_command(self, command):
        if not self.connected:
            return False
            
        try:
            message = json.dumps(command)
            self.socket.send(message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Send error: {e}")
            return False
    
    def get_game_state(self, timeout=1.0):
        try:
            return self.game_state_queue.get(timeout=timeout)
        except:
            return None
    
    def get_board_state(self, game_state):
        board_flat = game_state.get('board', [])
        if len(board_flat) == 200:  # 10x20 board
            return np.array(board_flat).reshape(20, 10)
        else:
            height = len(board_flat) // 10
            return np.array(board_flat).reshape(height, 10)
    
    def get_current_piece_info(self, game_state):
        piece_info = game_state.get('currentPiece', [0, 0, 0, 0])
        return {
            'type': piece_info[0],
            'rotation': piece_info[1],
            'x': piece_info[2],
            'y': piece_info[3]
        }
    
    def action_to_column_rotation(self, action_index):
        column = action_index // 4
        rotation = action_index % 4
        return column, rotation
    
    def disconnect(self):
        self.running = False
        self.connected = False
        if self.socket:
            self.socket.close()