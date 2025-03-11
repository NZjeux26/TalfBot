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
    
    # Count Black pieces before the move
    black_pieces_before = np.sum(initial_layout[game.BLACK])
    
    # Make the move to capture the black piece
    move = ((6, 4), (6, 2))  # White moves from a7 to b7 to capture black This is assuming this move is black when we are moving white to capture black instead
    new_state = game.make_move(initial_layout, move)
    new_state 
    
    # Print resulting board layout
    print("Resulting Board Layout:")
    print_board(new_state)
    
    # Count Black pieces after the move
    black_pieces_after = np.sum(new_state[game.BLACK])

    # Check if a Black piece was captured (count should decrease by one)
    print(f"Black pieces before: {black_pieces_before}, after: {black_pieces_after}")
    assert black_pieces_after == black_pieces_before - 1, "A Black piece should have been captured!"

if __name__ == '__main__':
    pytest.main()