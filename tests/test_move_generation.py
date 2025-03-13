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
from converttomatrix import initialise_board, print_board

@pytest.fixture
def game():
    return HnefataflGame(policy_value_net=None)

@pytest.fixture
def initial_layout(game):
    layout = initialise_board()
    
    #Clear the board of all pieces
    layout[game.BLACK] = np.zeros((11, 11), dtype=np.float32)
    layout[game.WHITE] = np.zeros((11, 11), dtype=np.float32)
    layout[game.KING] = np.zeros((11, 11), dtype=np.float32)
    
    return layout

#Tests white capturing black in a vertical line
def test_white_capture_black_TB(game, initial_layout):
    
    # Modify the initial layout for the test scenario
    initial_layout[game.BLACK, 7, 2] = 1  # Black piece at b7
    initial_layout[game.WHITE, 8, 2] = 1  # White piece at a7
    initial_layout[game.WHITE, 6, 4] = 1  # White piece at e5
    
    # Ensure it's White's turn
    initial_layout[game.PLAYER] = np.ones_like(initial_layout[game.PLAYER])
    
    #Print initial board layout
    print("Initial Board Layout:")
    print_board(initial_layout)
    
    # Count Black pieces before the move
    black_pieces_before = np.sum(initial_layout[game.BLACK])
    
    #Make the move to capture the black piece
    move = ((6, 4), (6, 2))  # White moves from a7 to b7 to capture black
    new_state = game.make_move(initial_layout, move)
    
    # Print resulting board layout
    print("Resulting Board Layout:")
    print_board(new_state)
    
    # Count Black pieces after the move
    black_pieces_after = np.sum(new_state[game.BLACK])

    # Check if a Black piece was captured (count should decrease by one)
    print(f"Black pieces before: {black_pieces_before}, after: {black_pieces_after}")
    assert black_pieces_after == black_pieces_before - 1, "A Black piece should have been captured!"

#Tests black capturing white in a vertical line
def test_black_capture_white_TB(game, initial_layout):
    
    # Modify the initial layout for the test scenario
    initial_layout[game.WHITE, 7, 2] = 1  # Black piece at b7
    initial_layout[game.BLACK, 8, 2] = 1  # White piece at a7
    initial_layout[game.BLACK, 6, 4] = 1  # White piece at e5
    
    #Print initial board layout
    print("Initial Board Layout:")
    print_board(initial_layout)
    
    # Count Black pieces before the move
    white_pieces_before = np.sum(initial_layout[game.WHITE])
    
    #Make the move to capture the black piece
    move = ((6, 4), (6, 2))  # White moves from a7 to b7 to capture black
    new_state = game.make_move(initial_layout, move)
    
    # Print resulting board layout
    print("Resulting Board Layout:")
    print_board(new_state)
    
    # Count Black pieces after the move
    white_pieces_after = np.sum(new_state[game.WHITE])

    # Check if a Black piece was captured (count should decrease by one)
    print(f"Black pieces before: {white_pieces_before}, after: {white_pieces_after}")
    assert white_pieces_after == white_pieces_before - 1, "A Black piece should have been captured!"
    
if __name__ == '__main__':
    pytest.main()