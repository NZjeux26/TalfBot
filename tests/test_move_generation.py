import pytest
import numpy as np
import sys
import os

# Get the absolute path to your project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add the absolute paths
sys.path.append(os.path.join(project_root, "Data_Conversion_and_Validation"))
sys.path.append(os.path.join(project_root, "Training"))
sys.path.append(os.path.join(project_root, "Self_Training"))

from Move_Generation import HnefataflGame
from converttomatrix import initialize_board, print_board

@pytest.fixture
def game():
    return HnefataflGame(policy_value_net=None)

@pytest.fixture
def initial_layout(game):
    layout = initialize_board()
    # Modify the initial layout for the test scenario
    layout[game.BLACK, 7, 2] = 1  # Black piece at b7
    layout[game.WHITE, 8, 2] = 1  # White piece at a7
    
    # Ensure it's White's turn
    layout[game.PLAYER] = np.ones_like(layout[game.PLAYER])
    
    return layout

def test_simple_capture(game, initial_layout):
    # Print initial board layout
    print("Initial Board Layout:")
    print_board(initial_layout)
    
    # Make the move to capture the black piece
    move = ((6, 4), (6, 2))  # White moves from a7 to b7 to capture black This is assuming this move is black when we are moving white to capture black instead
    new_state = game.make_move(initial_layout, move)
    
    # Print resulting board layout
    print("Resulting Board Layout:")
    print_board(new_state)
    
    # Check if the black piece was captured using _get_captures
    player_moving = game.WHITE  # Ensure we pass White as the player making the move
    captures = game._get_captures(new_state, (7, 2), player_moving) #This isn't right, I don't think I should be calling .Get_captures. 
            
    print("Captures found:", captures)
    assert (6, 2) in captures

if __name__ == '__main__':
    pytest.main()