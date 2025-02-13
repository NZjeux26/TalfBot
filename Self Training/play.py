import os
import sys
import torch
import numpy as np
from copy import deepcopy

# Get the absolute path to your project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the absolute paths
sys.path.append(os.path.join(project_root, "Data_Conversion_and_Validation"))
sys.path.append(os.path.join(project_root, "Training"))

# Import your neural network architecture
from model import PolicyValueNetwork
from MCTSTwo import MCTS
from converttomatrix import process_move
from Move_Generation import HnefataflGame

class ModelPlayer:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = PolicyValueNetwork()
        # Use weights_only=True to address the warning
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
      
    def get_policy_value(self, state):
        """Convert state to tensor and get model predictions"""
        with torch.no_grad():
            # Reshape state to [channels, height, width] if needed
            if state.shape[0] == 1:  # If first dimension is 1, squeeze it
                state = state.squeeze(0)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            start_probs, end_probs, value = self.model(state_tensor)
            
            # Convert outputs to numpy arrays
            start_probs = start_probs[0].cpu().numpy()
            end_probs = end_probs[0].cpu().numpy()
            value = value[0].item()
            
            return start_probs, end_probs, value

def play_interactive_game():
    # Initialize components
    model_player = ModelPlayer("hnefatafl_policy_value_model.pth")
    game = HnefataflGame(model_player.get_policy_value)
    
    # Initialize MCTS
    mcts = MCTS(game, n_simulations=1600, c_puct=1.0)
    
    # Game settings
    human_player = input("Do you want to play as black or white? ").lower()
    temperature = 1.0
    
    state = game.get_initial_state()
    move_count = 0
    
    print("\nGame starting!")
    print("Enter moves in algebraic notation (e.g., 'f6-f4')")
    print("Type 'quit' to end the game\n")
    
    while True:
        game.print_state(state)
        current_player = 'white' if state[game.PLAYER].sum() > 0 else 'black'
        
        # Get model's evaluation of position
        _, _, value = game.get_policy_value_predictions(state) #issue is in here
        #policy_probs, value = game.get_policy_value_predictions(state)
        print(f"\nModel evaluation: {value:.3f}")
        
        # Get valid moves for current position
        valid_moves = game.get_valid_moves(state)
        if not valid_moves:
            print(f"No valid moves available for {current_player}!")
            game_result = 1 if current_player == 'black' else -1
            break
        
        try:
            if current_player == human_player:
                # Human move
                while True:
                    move_str = input(f"\nYour move ({current_player}): ")
                    if move_str.lower() == 'quit':
                        return
                        
                    start, end, _ = process_move(move_str)
                    if start is None or end is None:
                        print("Invalid move format. Use 'e6-e4' format.")
                        continue
                        
                    if (start, end) in valid_moves:
                        move = (start, end)
                        break
                    print("Invalid move, try again.")
            else:
                # Model move
                print(f"\nModel ({current_player}) is thinking...")
                action_probs = mcts.get_action_probs(state, temperature)
                moves, probs = zip(*action_probs.items())
                move = moves[np.argmax(probs)]
                
                # Convert move to algebraic notation for display
                start_alg = chr(ord('a') + move[0][1]) + str(11 - move[0][0])
                end_alg = chr(ord('a') + move[1][1]) + str(11 - move[1][0])
                print(f"Model plays: {start_alg}-{end_alg}")
            
            # Make the move
            state = game.make_move(state, move)
            move_count += 1
            
            # Check for game end
            game_result = game.get_game_ended(state)
            if game_result != 0:
                game.print_state(state)
                winner = "White" if game_result == 1 else "Black"
                print(f"\nGame Over! {winner} wins in {move_count} moves!")
                break
            
            # Optional: adjust temperature as game progresses
            if move_count == 10:
                temperature = 0.75
            elif move_count == 20:
                temperature = 0.5
                
        except Exception as e:
            print(f"An error occurred: {e}")
            return #debugging to break loop.
            print("Please try again.")
            
if __name__ == "__main__":
    print("Welcome to Hnefatafl!")
    print("=" * 50)
    try:
        play_interactive_game()
    except KeyboardInterrupt:
        print("\nGame terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nThanks for playing!")