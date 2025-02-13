import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
# Get the absolute path to your project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the absolute paths
sys.path.append(os.path.join(project_root, "Data_Conversion_and_Validation"))

from converttomatrix import initialize_board, update_history_channel, print_board

class HnefataflGame:
    # Channel indices
    BLACK, WHITE, KING, SPECIAL, HISTORY, PLAYER = 0, 1, 2, 3, 4, 5
    BOARD_SIZE = 11
    
    def __init__(self, policy_value_net):
        self.policy_value_net = policy_value_net
        self.state = self.get_initial_state()
    
    def get_initial_state(self):
        """Initialize the 6-channel board state"""
        return initialize_board()
    
    def get_policy_value_predictions(self, state):
        """Get start position, end position, and value predictions from the network"""
        with torch.no_grad():
            network_input = torch.FloatTensor(state).unsqueeze(0)
            start_probs, end_probs, value = self.policy_value_net(network_input)
            
            # Convert to numpy arrays
            start_probs = start_probs.squeeze(0).numpy()  # Shape: (121,)
            end_probs = end_probs.squeeze(0).numpy()      # Shape: (121,)
            value = value.item()
            
            return start_probs, end_probs, value
    
    def get_move_probabilities(self, state, valid_moves):
        """
        Convert network outputs to valid move probabilities
        Returns: dict mapping moves (start, end) to their probabilities
        """
        start_probs, end_probs, _ = self.get_policy_value_predictions(state)
        move_probs = {}
        
        for start, end in valid_moves:
            # Convert 2D positions to 1D indices
            start_idx = start[0] * self.BOARD_SIZE + start[1]
            end_idx = end[0] * self.BOARD_SIZE + end[1]
            
            # Combine probabilities for the move
            move_prob = start_probs[start_idx] * end_probs[end_idx]
            move_probs[(start, end)] = move_prob
        
        # Normalize probabilities
        if move_probs:
            total_prob = sum(move_probs.values())
            if total_prob > 0:
                move_probs = {k: v/total_prob for k, v in move_probs.items()}
            else:
                # If all probabilities are zero, use uniform distribution
                uniform_prob = 1.0 / len(valid_moves)
                move_probs = {k: uniform_prob for k in move_probs}
        
        return move_probs
    
    def encode_move(self, move):
        """Convert move tuple to network input format"""
        start, end = move
        start_idx = start[0] * self.BOARD_SIZE + start[1]
        end_idx = end[0] * self.BOARD_SIZE + end[1]
        return start_idx, end_idx
    
    def decode_move(self, start_idx, end_idx):
        """Convert network output indices to move tuple"""
        start = (start_idx // self.BOARD_SIZE, start_idx % self.BOARD_SIZE)
        end = (end_idx // self.BOARD_SIZE, end_idx % self.BOARD_SIZE)
        return start, end
    
    def get_valid_moves(self, state):
        """Get all valid moves for current player with uniform probabilities"""
        is_black_turn = (state[self.PLAYER].sum() == 0)
        valid_moves = []
        
        # Debug print
        print(f"Checking moves for {'black' if is_black_turn else 'white'}")
        
        # Get pieces for current player
        if is_black_turn:
            piece_positions = np.where(state[self.BLACK] == 1)
            print(f"Found {len(piece_positions[0])} black pieces")
        else:
            white_positions = np.where(state[self.WHITE] == 1)
            king_positions = np.where(state[self.KING] == 1)
            piece_positions = (
                np.concatenate([white_positions[0], king_positions[0]]),
                np.concatenate([white_positions[1], king_positions[1]])
            )
            print(f"Found {len(piece_positions[0])} white pieces (including king)")
            
        # Check all possible moves for each piece
        for i in range(len(piece_positions[0])):
            start = (piece_positions[0][i], piece_positions[1][i])
            
            # Check horizontal moves
            for col in range(self.BOARD_SIZE):
                if self._is_valid_move(state, start, (start[0], col)):
                    valid_moves.append((start, (start[0], col)))
            
            # Check vertical moves
            for row in range(self.BOARD_SIZE):
                if self._is_valid_move(state, start, (row, start[1])):
                    valid_moves.append((start, (row, start[1])))
                    
        print(f"Generated {len(valid_moves)} valid moves")
        # Convert to dictionary with uniform probabilities
        if valid_moves:
            uniform_prob = 1.0 / len(valid_moves)
            return {move: uniform_prob for move in valid_moves}
        return {}
    
    def get_policy_value_predictions(self, state):
        """Get both policy and value predictions from the network"""
        with torch.no_grad():
            network_input = torch.FloatTensor(state).unsqueeze(0)
            policy_output, value_output = self.policy_value_net(network_input)
            
            # Convert policy output to probabilities
            policy_probs = F.softmax(policy_output, dim=1).squeeze(0).numpy()
            value = value_output.item()
            
            return policy_probs, value
    
    def _is_valid_move(self, state, start, end):
        """Check if move is valid"""
        if start == end:
            return False
            
        # Check if destination is occupied
        if (state[self.BLACK:self.KING+1, end[0], end[1]] > 0).any():
            return False
            
        # Check if moving piece belongs to current player
        is_black_turn = (state[self.PLAYER].sum() == 0)
        if is_black_turn and state[self.BLACK, start[0], start[1]] != 1:
            return False
        if not is_black_turn and state[self.BLACK, start[0], start[1]] == 1:
            return False
            
        # Special rules for throne and corners
        if state[self.SPECIAL, end[0], end[1]] == 1:
            # Only king can enter corners and throne
            if state[self.KING, start[0], start[1]] != 1:
                return False
        
        # Check for clear path
        if start[0] == end[0]:  # Horizontal move
            min_col = min(start[1], end[1])
            max_col = max(start[1], end[1])
            for col in range(min_col + 1, max_col):
                if (state[self.BLACK:self.KING+1, start[0], col] > 0).any():
                    return False
        else:  # Vertical move
            min_row = min(start[0], end[0])
            max_row = max(start[0], end[0])
            for row in range(min_row + 1, max_row):
                if (state[self.BLACK:self.KING+1, row, start[1]] > 0).any():
                    return False
        
        return True
    
    def make_move(self, state, move):
        """Make move and return new state with captures"""
        new_state = state.copy()
        start, end = move
        
        # Move piece
        for channel in [self.BLACK, self.WHITE, self.KING]:
            if state[channel, start[0], start[1]] == 1:
                new_state[channel, start[0], start[1]] = 0
                new_state[channel, end[0], end[1]] = 1
                break
        
        # Handle captures
        captures = self._get_captures(new_state, end)
        for capture_pos in captures:
            for channel in [self.BLACK, self.WHITE, self.KING]:
                if new_state[channel, capture_pos[0], capture_pos[1]] == 1:
                    new_state[channel, capture_pos[0], capture_pos[1]] = 0
        
        # Update history channel
        new_state = update_history_channel(new_state, start, end, 
                                         capture_pos=captures[0] if captures else None)
        
        # Switch player
        new_state[self.PLAYER] = 1 - new_state[self.PLAYER]
        
        return new_state
    
    def _get_captures(self, state, pos):
        """Get all pieces captured by the move"""
        captures = []
        is_black = (state[self.PLAYER].sum() == 0)  # Current player before switch
        
        # Check all four directions
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            capture_pos = self._check_capture(state, pos, (dx, dy), is_black)
            if capture_pos:
                captures.append(capture_pos)
        
        return captures
    
    def get_game_ended(self, state):
        """Check if game is over and return winner"""
        # Check if king reached corner
        king_pos = np.where(state[self.KING] == 1)
        if len(king_pos[0]) == 0:  # King captured
            return 1 if state[self.PLAYER].sum() == 0 else -1
        
        if len(king_pos[0]) > 0:
            kr, kc = king_pos[0][0], king_pos[1][0]
            if state[self.SPECIAL, kr, kc] == 1 and (kr in [0, 10] or kc in [0, 10]):
                return -1 if state[self.PLAYER].sum() == 0 else 1
        
        return 0  # Game not ended
    
    def _check_capture(self, state, pos, direction, is_black):
        """
        Check for captures using Copenhagen rules:
        - Regular pieces need 2 attackers or 1 attacker + throne/corner/hostile piece
        - King needs 4 attackers (3 if against throne/corner)
        - Captures are custodial (pieces must be adjacent)
        
        Args:
            state: Current game state
            pos: Position of the piece that just moved (x, y)
            direction: Direction to check (dx, dy)
            is_black: Boolean indicating if current player is black
        
        Returns:
            Position of captured piece if capture occurs, None otherwise
        """
        x, y = pos
        dx, dy = direction
        board_size = state.shape[-1]
        
        # Position to check for potential capture
        capture_x = x + dx
        capture_y = y + dy
        
        # Check if capture position is within bounds
        if not (0 <= capture_x < board_size and 0 <= capture_y < board_size):
            return None
            
        # Get piece at capture position
        capture_piece = None
        is_king = False
        if state[self.BLACK, capture_x, capture_y] == 1:
            capture_piece = 'black'
        elif state[self.WHITE, capture_x, capture_y] == 1:
            capture_piece = 'white'
        elif state[self.KING, capture_x, capture_y] == 1:
            capture_piece = 'king'
            is_king = True
        
        # If no piece to capture or piece is same color as current player
        if capture_piece is None:
            return None
        if (is_black and capture_piece == 'black') or (not is_black and capture_piece in ['white', 'king']):
            return None
        
        # Define special squares
        throne = (board_size // 2, board_size // 2)
        corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
        
        # Count attackers around the captured piece
        attackers = 0
        hostile_squares = 0
        
        for check_dir in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            check_x = capture_x + check_dir[0]
            check_y = capture_y + check_dir[1]
            
            # Skip if out of bounds
            if not (0 <= check_x < board_size and 0 <= check_y < board_size):
                if is_king:
                    hostile_squares += 1
                continue
                
            # Check if square contains an attacker
            if is_black:
                is_attacker = state[self.BLACK, check_x, check_y] == 1
            else:
                is_attacker = (state[self.WHITE, check_x, check_y] == 1 or 
                            state[self.KING, check_x, check_y] == 1)
                
            if is_attacker:
                attackers += 1
                
            # Check for hostile squares (throne, corners)
            if (check_x, check_y) == throne and not state[self.KING, throne[0], throne[1]]:
                hostile_squares += 1
            elif (check_x, check_y) in corners:
                hostile_squares += 1
                
        # Apply Copenhagen capture rules
        if is_king:
            # King needs 4 attackers, or 3 if against hostile square
            required_attackers = 4 - hostile_squares
            if attackers >= required_attackers and required_attackers >= 3:
                return (capture_x, capture_y)
        else:
            # Regular pieces need 2 attackers or 1 attacker + hostile square
            if attackers + hostile_squares >= 2:
                return (capture_x, capture_y)
                
        return None
    
    def print_state(self, state):
        # Use your existing print_board function
        print_board(state)
        print(f"Current player: {'White' if state[self.PLAYER].sum() > 0 else 'Black'}")