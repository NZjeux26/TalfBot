import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
# Get the absolute path to your project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the absolute paths
sys.path.append(os.path.join(project_root, "Data_Conversion_and_Validation"))

from converttomatrix import initialise_board, update_history_channel, print_board

class HnefataflGame:
    # Channel indices
    BLACK, WHITE, KING, SPECIAL, HISTORY, PLAYER = 0, 1, 2, 3, 4, 5
    BOARD_SIZE = 11
    
    def __init__(self, policy_value_net):
        self.policy_value_net = policy_value_net
        self.state = self.get_initial_state()
    
    def get_initial_state(self):
        """Initialize the 6-channel board state"""
        return initialise_board()
    
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
    
    def get_valid_moves(self, state, print_once = False):
        """Get all valid moves for current player with uniform probabilities"""
        is_black_turn = (state[self.PLAYER].sum() == 0)
        
        # # Debug print
        # if print_once:
        #     print(f"Checking moves for {'black' if is_black_turn else 'white'}")
        
        valid_moves = []
        
        # Get pieces for current player
        if is_black_turn:
            piece_positions = np.where(state[self.BLACK] == 1)
            #print(f"Found {len(piece_positions[0])} black pieces")
        else:
            white_positions = np.where(state[self.WHITE] == 1)
            king_positions = np.where(state[self.KING] == 1)
            piece_positions = (
                np.concatenate([white_positions[0], king_positions[0]]),
                np.concatenate([white_positions[1], king_positions[1]])
            )
            #print(f"Found {len(piece_positions[0])} white pieces (including king)")
            
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
        return valid_moves
      
    
    def get_policy_value_predictions(self, state):
        """Get both policy and value predictions from the network."""
        with torch.no_grad():
            # Handle different state formats
            if len(state.shape) == 3 and state.shape[0] == 6:  # Expected [6, 11, 11]
                network_input = torch.FloatTensor(state).unsqueeze(0)
            else:
                network_input = torch.FloatTensor(state)
                if len(network_input.shape) == 3:
                    network_input = network_input.unsqueeze(0)

            # Pass through the policy-value network
            start_probs, end_probs, value_output = self.policy_value_net(network_input)

            # Convert outputs to numpy arrays with proper tensor handling
            if isinstance(start_probs, torch.Tensor):
                start_probs = start_probs.squeeze(0).cpu().numpy()
            else:
                start_probs = np.array(start_probs)
                
            if isinstance(end_probs, torch.Tensor):
                end_probs = end_probs.squeeze(0).cpu().numpy()
            else:
                end_probs = np.array(end_probs)

            # Ensure value is a scalar
            if isinstance(value_output, torch.Tensor):
                value = value_output.squeeze().cpu().item()
            else:
                value = float(value_output)

            return start_probs, end_probs, value

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
            capture_pos = self._check_capture(state, pos, (dx, dy), is_black) #checks every direction for a valid capture
            if capture_pos:
                captures.append(capture_pos)
        
        # Check for shield wall captures
        shield_wall_captures = self._check_shield_wall_capture(state, pos)
        captures.extend(shield_wall_captures)
        
        return captures
    
    def check_encirclement(self, state):
        """
        Check if the attackers (black pieces) have completely surrounded the king 
        and all defenders with an unbroken ring.
        
        Returns:
            bool: True if encirclement is complete, False otherwise
        """
        board_size = state.shape[-1]
        
        # Create a merged board showing all pieces
        # 0: empty, 1: black (attackers), 2: white/king (defenders)
        merged_board = np.zeros((board_size, board_size), dtype=np.int8)
        
        # Mark attackers as 1
        merged_board[np.where(state[self.BLACK] == 1)] = 1
        
        # Mark defenders (white pieces and king) as 2
        merged_board[np.where(state[self.WHITE] == 1)] = 2
        merged_board[np.where(state[self.KING] == 1)] = 2
        
        # Find all defenders
        defender_squares = np.where(merged_board == 2)
        defender_coords = list(zip(defender_squares[0], defender_squares[1]))
        
        if not defender_coords:
            return False  # No defenders left
        
        # Find all empty squares
        empty_squares = np.where(merged_board == 0)
        empty_coords = set(zip(empty_squares[0], empty_squares[1]))
        
        # Get corners and edges
        corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
        edges = set()
        for i in range(board_size):
            edges.add((0, i))
            edges.add((i, 0))
            edges.add((board_size-1, i))
            edges.add((i, board_size-1))
        
        # Start flood fill from any defender
        start_x, start_y = defender_coords[0]
        queue = [(start_x, start_y)]
        visited = set([(start_x, start_y)])
        can_reach_edge = False
        
        # Perform BFS flood fill
        while queue and not can_reach_edge:
            x, y = queue.pop(0)
            
            # Check if we reached an edge
            if (x, y) in edges:
                can_reach_edge = True
                break
            
            # Check all four adjacent squares
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                # Ensure we're within bounds
                if not (0 <= nx < board_size and 0 <= ny < board_size):
                    continue
                
                # If this is an empty square or defender we haven't visited yet
                if ((merged_board[nx, ny] == 0 or merged_board[nx, ny] == 2) and 
                    (nx, ny) not in visited):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        
        # If defenders can't reach an edge, they're completely surrounded
        return not can_reach_edge

    def check_move_repetition(self, move_history, current_move):
        """
        Check for perpetual repetition based on move sequences.
        Tracks recent moves and detects repetitive patterns.
        
        Args:
            move_history: List of recent moves [(start1, end1), (start2, end2), ...]
            current_move: The move just made (start, end)
            
        Returns:
            True if repetition is detected (3 identical sequences), None otherwise
        """
        # Add the current move to history
        if current_move not in move_history:
            move_history.append(current_move)
        
        # Limit history size to prevent excessive memory usage
        if len(move_history) > 30:  # Keep only the last 30 moves
            move_history = move_history[-30:]
        
        # Look for sequences of 4 moves (2 moves by each player)
        seq_length = 4
        
        if len(move_history) >= 3 * seq_length:
            # Check the last sequence
            latest_seq = move_history[-seq_length:]
            
            # Check if this sequence appears 3 times in total
            count = 0
            for i in range(0, len(move_history) - seq_length + 1, seq_length):
                current_seq = move_history[i:i+seq_length]
                if current_seq == latest_seq:
                    count += 1
            
            if count >= 3:
                return True
        
        return None
    
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
        
        # Check for encirclement win condition
        if self.check_encirclement(state):
            return 1
    
        return 0  # Game not ended
    
    def _check_shield_wall_capture(self, state, pos):
        """
        Check for shield wall captures according to Copenhagen rules:
        - A row of 2+ pieces along the board edge can be captured together
        - Both ends of the row must be bracketed by enemy pieces or corners
        - Each piece in the wall must have an enemy piece directly in front of it

        Args:
            state: Current game state
            pos: Position of the piece that just moved (x, y)

        Returns:
            List of positions of captured pieces if shield wall capture occurs, empty list otherwise
        """
        x, y = pos
        board_size = state.shape[-1]
        is_black = (state[self.PLAYER].sum() == 0)  # Current player before switch

        # Only check for shield wall if the piece that moved is at the edge but not corner
        if not ((x == 0 or x == board_size-1 or y == 0 or y == board_size-1) and 
                not ((x == 0 and y == 0) or (x == 0 and y == board_size-1) or 
                        (x == board_size-1 and y == 0) or (x == board_size-1 and y == board_size-1))):
            return []

        # Determine which edge we're on
        on_top = (x == 0)
        on_bottom = (x == board_size-1)
        on_left = (y == 0)
        on_right = (y == board_size-1)

        # Determine the direction to check for opposing pieces (perpendicular to the edge)
        front_dir = None
        if on_top:
            front_dir = (1, 0)  # Check downward
        elif on_bottom:
            front_dir = (-1, 0)  # Check upward
        elif on_left:
            front_dir = (0, 1)  # Check rightward
        elif on_right:
            front_dir = (0, -1)  # Check leftward

        # Determine the direction along the edge
        edge_dir = None
        if on_top or on_bottom:
            edge_dir = [(0, 1), (0, -1)]  # Check horizontally
        else:
            edge_dir = [(1, 0), (-1, 0)]  # Check vertically

        # Define corner squares
        corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]

        # Check both directions along the edge for potential shield walls
        captured_positions = []

        for dx, dy in edge_dir:
            # Find continuous rows of enemy pieces along the edge
            wall_pieces = []
            
            # Start at current position and extend in current direction
            curr_x, curr_y = x, y
            
            # Keep track of attacker positions at the ends
            attacker_positions = []  # Will store tuples of (position, is_corner)
            
            # Check if the current position is an attacker
            current_is_attacker = (is_black and state[self.BLACK, x, y] == 1) or \
                                    (not is_black and (state[self.WHITE, x, y] == 1 or state[self.KING, x, y] == 1))
            
            if current_is_attacker:
                attacker_positions.append(((x, y), False))
            
            # Search along the edge
            while True:
                curr_x += dx
                curr_y += dy
                
                # Check if we're still in bounds and on the edge
                if not (0 <= curr_x < board_size and 0 <= curr_y < board_size):
                    break
                    
                # Check if we hit a corner (can be used as a bracketing piece)
                if (curr_x, curr_y) in corners:
                    attacker_positions.append(((curr_x, curr_y), True))
                    break
                    
                # Check what piece is at this position
                is_black_piece = state[self.BLACK, curr_x, curr_y] == 1
                is_white_piece = state[self.WHITE, curr_x, curr_y] == 1
                is_king = state[self.KING, curr_x, curr_y] == 1
                
                # Is this position part of the shield wall (enemy piece)?
                is_enemy = (is_black and (is_white_piece or is_king)) or \
                            (not is_black and is_black_piece)
                
                # Is this position an attacker (friendly piece)?
                is_attacker = (is_black and is_black_piece) or \
                                (not is_black and (is_white_piece or is_king))
                
                # If empty space, break
                if not (is_enemy or is_attacker):
                    break
                    
                # If we found an enemy piece, add to the wall
                if is_enemy:
                    # First, check if it has an attacker in front of it
                    front_x = curr_x + front_dir[0]
                    front_y = curr_y + front_dir[1]
                    
                    # Check if front position is in bounds
                    if not (0 <= front_x < board_size and 0 <= front_y < board_size):
                        break
                    
                    # Check if there's an attacker in front
                    has_front_attacker = (is_black and state[self.BLACK, front_x, front_y] == 1) or \
                                        (not is_black and (state[self.WHITE, front_x, front_y] == 1 or 
                                                        state[self.KING, front_x, front_y] == 1))
                    
                    if has_front_attacker:
                        wall_pieces.append((curr_x, curr_y))
                    else:
                        # No attacker in front, shield wall condition fails
                        break
                
                # If we found another attacker, add to bracketing pieces
                if is_attacker:
                    attacker_positions.append(((curr_x, curr_y), False))
                    break
            
            # Now check if we have a valid shield wall
            if len(wall_pieces) >= 2 and len(attacker_positions) >= 2:
                # Special rule: If king is part of wall, don't capture him
                if not is_black:  # If white player (defenders) is moving
                    # Filter out king positions from captures
                    wall_pieces = [(x, y) for x, y in wall_pieces if not state[self.KING, x, y]]
                
                captured_positions.extend(wall_pieces)

        return captured_positions

    def _check_capture(self, state, pos, direction, is_black):
        """
        Check for captures using Copenhagen rules:
        - Regular pieces need 2 attackers or 1 attacker + throne/corner/hostile piece
        - King needs 4 attackers (3 if against throne/corner)
        - Captures are custodial (pieces must be adjacent and on opposite sides)
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
        
        # The piece that just moved is an attacker
        attackers = 1
    
    # Check if there's another attacker on the opposite side (for sandwiching)
        opposite_x = capture_x + dx
        opposite_y = capture_y + dy
        
        # Ensure the opposite position is within bounds
        if not (0 <= opposite_x < board_size and 0 <= opposite_y < board_size):
            opposite_attacker = False
        else:
            # Check if the opposite square contains an attacker
            if is_black:
                opposite_attacker = state[self.BLACK, opposite_x, opposite_y] == 1
            else:
                opposite_attacker = (state[self.WHITE, opposite_x, opposite_y] == 1 or 
                            state[self.KING, opposite_x, opposite_y] == 1)
        
        # Check if the opposite position is a hostile square (throne/corners)
        is_opposite_hostile = ((opposite_x, opposite_y) == throne and not state[self.KING, throne[0], throne[1]]) or \
                            ((opposite_x, opposite_y) in corners)
        
        # For non-king pieces, we need either:
        # 1. Two attackers sandwiching the piece, or
        # 2. One attacker + a hostile square
        if not is_king:
            if (opposite_attacker or is_opposite_hostile):
                return (capture_x, capture_y)
            return None
        
        # For the king, special rules apply
        # Count additional attackers and hostile squares
        additional_attackers = 0
        hostile_squares = 0
        
        # Check the other sides (perpendicular to the direction of movement)
        perp_dirs = []
        if dx != 0:  # If moving horizontally, check vertical positions
            perp_dirs = [(0, 1), (0, -1)]
        else:  # If moving vertically, check horizontal positions
            perp_dirs = [(1, 0), (-1, 0)]
        
        for perp_dx, perp_dy in perp_dirs:
            adj_x = capture_x + perp_dx
            adj_y = capture_y + perp_dy
            
            # Ensure the position is within bounds
            if not (0 <= adj_x < board_size and 0 <= adj_y < board_size):
                continue
            
            # Check if the square contains an attacker
            if is_black:
                is_adj_attacker = state[self.BLACK, adj_x, adj_y] == 1
            else:
                is_adj_attacker = (state[self.WHITE, adj_x, adj_y] == 1 or 
                        state[self.KING, adj_x, adj_y] == 1)
            
            # Check if the position is a hostile square
            is_adj_hostile = ((adj_x, adj_y) == throne and not state[self.KING, throne[0], throne[1]]) or \
                        ((adj_x, adj_y) in corners)
            
            if is_adj_attacker:
                additional_attackers += 1
            elif is_adj_hostile:
                hostile_squares += 1
        
        # Apply Copenhagen capture rules for king
        total_attackers = attackers + additional_attackers + (1 if opposite_attacker else 0)
        total_hostiles = hostile_squares + (1 if is_opposite_hostile else 0)
        
        # King needs 4 attackers or fewer if against hostile squares
        required_attackers = 4 - total_hostiles
        if total_attackers >= required_attackers and required_attackers >= 3:
            return (capture_x, capture_y)
        
        return None
    
    def print_state(self, state):
        # Use your existing print_board function
        print_board(state)
        print(f"Current player: {'White' if state[self.PLAYER].sum() > 0 else 'Black'}")