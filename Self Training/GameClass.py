import numpy as np
from converttomatrix import initialize_board, print_board, process_move #cannot see the 
from MCTSTwo import MCTS
from copy import deepcopy

class HnefataflGame:
    def __init__(self):
        # Constants from your conversion code
        self.BLACK, self.WHITE, self.KING, self.SPECIAL, self.HISTORY, self.PLAYER = 0, 1, 2, 3, 4, 5
        
    def get_initial_state(self):
        # Use your existing initialization
        return initialize_board()
    
    def get_valid_moves(self, state):
        # This should return a dict of valid moves
        # You'll need to adapt your move generation code to return:
        # {(start_pos, end_pos): probability}
        # For now, we can use uniform probabilities
        valid_moves = {}  # Your move generation code here
        if valid_moves:
            uniform_prob = 1.0 / len(valid_moves)
            return {move: uniform_prob for move in valid_moves}
        return {}
    
    def make_move(self, state, move):
        # Create a copy of the state
        new_state = deepcopy(state)
        
        start_pos, end_pos = move
        # Use your existing move application logic
        # Move the piece
        for channel in [self.BLACK, self.WHITE, self.KING]:
            if new_state[channel, start_pos[0], start_pos[1]] == 1:
                new_state[channel, start_pos[0], start_pos[1]] = 0
                new_state[channel, end_pos[0], end_pos[1]] = 1
                break
        
        # Handle captures (your existing capture logic)
        # Update history channel using your update_history_channel function
        
        # Switch player
        new_state[self.PLAYER] = 1 - new_state[self.PLAYER]
        
        return new_state
    
    def get_game_ended(self, state):
        # Check win conditions
        # Return:
        #  1 for white win
        # -1 for black win
        #  0 for ongoing game
        
        # Check king escape (corners)
        corners = [(0, 0), (0, 10), (10, 0), (10, 10)]
        king_channel = state[self.KING]
        for corner in corners:
            if king_channel[corner[0], corner[1]] == 1:
                return 1  # White wins
        
        # Check king capture
        # Your king capture detection logic here
        
        # Check if no valid moves (stalemate)
        if not self.get_valid_moves(state):
            return -1 if state[self.PLAYER].sum() == 0 else 1
            
        return 0  # Game ongoing
    
    def get_policy_value_predictions(self, state):
        # Use your trained network here
        # Should return:
        # - start position probabilities
        # - end position probabilities 
        # - value prediction (-1 to 1)
        # For now, return placeholders
        return (
            np.ones((11, 11)) / 121,  # Uniform start probabilities
            np.ones((11, 11)) / 121,  # Uniform end probabilities
            0.0  # Neutral value prediction
        )
    
    def print_state(self, state):
        # Use your existing print_board function
        print_board(state)
        print(f"Current player: {'White' if state[self.PLAYER].sum() > 0 else 'Black'}")

def play_game(game, model, human_player='black'):
    """
    Play a game against the model.
    
    Args:
        game: HnefataflGame instance
        model: Your trained neural network
        human_player: 'black' or 'white'
    """
    state = game.get_initial_state()
    mcts = MCTS(game, n_simulations=800)
    
    while True:
        game.print_state(state)
        
        current_player = 'white' if state[game.PLAYER].sum() > 0 else 'black'
        
        if current_player == human_player:
            # Human move
            while True:
                try:
                    move_str = input(f"Enter your move (e.g., 'f6-f4'): ")
                    start, end, _ = process_move(move_str)
                    if (start, end) in game.get_valid_moves(state):
                        break
                    print("Invalid move, try again.")
                except:
                    print("Invalid input format, try again.")
            move = (start, end)
        else:
            # Model move
            action_probs = mcts.get_action_probs(state, temperature=1)
            moves, probs = zip(*action_probs.items())
            move = moves[np.argmax(probs)]
            print(f"Model plays: {move}")
        
        state = game.make_move(state, move)
        
        game_result = game.get_game_ended(state)
        if game_result != 0:
            game.print_state(state)
            winner = "White" if game_result == 1 else "Black"
            print(f"Game Over! {winner} wins!")
            break