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

#Tests capturing in a vertical line for both teams
def test_capture_vertical(game, initial_layout):
    
    # Modify the initial layout for the test scenario Black
    initial_layout[game.WHITE, 7, 2] = 1  # Black piece at b7
    initial_layout[game.BLACK, 8, 2] = 1  # White piece at a7
    initial_layout[game.BLACK, 6, 4] = 1  # White piece at e5
    
    # Modify the initial layout for the test scenario White
    initial_layout[game.BLACK, 7, 7] = 1  # Black piece at b7
    initial_layout[game.WHITE, 8, 7] = 1  # White piece at a7
    initial_layout[game.WHITE, 6, 9] = 1  # White piece at e5
    
    #Print initial board layout
    print("Initial Board Layout:")
    print_board(initial_layout)
    
    # Count Black pieces before the move
    pieces_before = np.sum(initial_layout[game.WHITE]) + np.sum(initial_layout[game.BLACK])
    
    #Make the move to capture the black piece
    move = ((6, 4), (6, 2))  # White moves from a7 to b7 to capture black
    new_state = game.make_move(initial_layout, move)
    
    move = ((6, 9), (6, 7))  # White moves from a7 to b7 to capture black
    new_state = game.make_move(new_state, move)
    
    # Print resulting board layout
    print("Resulting Board Layout:")
    print_board(new_state)
    
    # Count pieces after the move
    pieces_after = np.sum(new_state[game.WHITE]) + np.sum(new_state[game.BLACK])

    # Check if a piece was captured (count should decrease by two)
    print(f"Black pieces before: {pieces_before}, after: {pieces_after}")
    assert pieces_after == pieces_before - 2, "A piece should have been captured!"

#This function tests capturing pieces against hostile squares (Corners and Empty Throne)
def test_capture_hostile_squares(game, initial_layout):
    
    #setup corner test positions for corner test
    initial_layout[game.BLACK, 0, 1] = 1 #Black piece in B11
    initial_layout[game.WHITE, 1, 2] = 1 #White piece in C10
    
    #setup for capturing against the throne
    initial_layout[game.WHITE, 4, 5] = 1 #White piece in F7
    initial_layout[game.BLACK, 2, 5] = 1 #Black in F9
    
    #Print initial board layout
    print("Initial Board Layout:")
    print_board(initial_layout)
    
    # Count all pieces before the move. We use both Black and white since after each play the colour in play switches in make_move
    pieces_before = np.sum(initial_layout[game.WHITE]) + np.sum(initial_layout[game.BLACK])
    
    #move to caoture a piece that is against the emtpy throne
    move = ((2, 5), (3, 5))  # move black from F7 to C8
    new_state = game.make_move(initial_layout, move)
    
    #Make the move to capture the black piece in the corner
    move = ((1, 2), (0, 2))  # move white from c10 -> C11
    new_state = game.make_move(new_state, move) #We make the move from new_state since a piece has already been moved
    
    #Print resulting board layout
    print("Resulting Board Layout:")
    print_board(new_state)
    
    # Count pieces after the move, if the pieces have been taken they would have been removed from the baord and missing in the array
    pieces_after = np.sum(new_state[game.WHITE]) + np.sum(new_state[game.BLACK])

    # Check if a piece was captured (count should decrease by one)
    print(f"Game pieces before: {pieces_before}, after: {pieces_after}")
    assert pieces_after == pieces_before - 2, "A game piece should have been captured!"

#This function tests multicapture. Where if there are multiple pieces that can be caught in one move, they are all removed.
def test_multicapture(game, initial_layout):
    
    #set up the pieces
    initial_layout[game.WHITE, 4, 1] = 1
    initial_layout[game.WHITE, 4, 3] = 1
    
    initial_layout[game.BLACK, 4, 0] = 1
    initial_layout[game.BLACK, 3, 2] = 1
    initial_layout[game.BLACK, 4, 4] = 1
    
    #Print initial board layout
    print("Initial Board Layout:")
    print_board(initial_layout)
    
    pieces_before = np.sum(initial_layout[game.WHITE]) + np.sum(initial_layout[game.BLACK])
    
    #move to capture a piece that is against the emtpy throne
    move = ((3, 2), (4, 2))
    new_state = game.make_move(initial_layout, move)
    
    #Print resulting board layout
    print("Resulting Board Layout:")
    print_board(new_state)
    
    pieces_after = np.sum(new_state[game.WHITE]) + np.sum(new_state[game.BLACK])
    
    #Check if a piece was captured (count should decrease by one)
    print(f"Game pieces before: {pieces_before}, after: {pieces_after}")
    assert pieces_after == pieces_before - 2, "A game piece should have been captured!"
    
#add king capture, both in the open and against the throne
def test_king_capture(game, initial_layout):
    
    #Normal four sided capture
    initial_layout[game.BLACK, 2, 5] = 1
    initial_layout[game.BLACK, 3, 4] = 1
    initial_layout[game.BLACK, 4, 5] = 1
    initial_layout[game.BLACK, 3, 7] = 1
    
    initial_layout[game.KING, 3, 5] = 1
    
    #King capture against Throne
    initial_layout[game.BLACK, 6, 4] = 1
    initial_layout[game.BLACK, 6, 6] = 1
    initial_layout[game.BLACK, 8, 5] = 1
    
    initial_layout[game.KING, 6, 5] = 1
    
    #Print initial board layout
    print("Initial Board Layout:")
    print_board(initial_layout)
    
    pieces_before = np.sum(initial_layout[game.KING])
    
    move = ((3, 7), (3, 6)) 
    new_state = game.make_move(initial_layout, move)
    
    new_state[game.PLAYER] = np.zeros_like(new_state[game.PLAYER]) #force Black's turn
    
    #move to capture the king against the emtpy throne
    move = ((8, 5), (7, 5)) 
    new_state = game.make_move(new_state, move)
    
    #Print resulting board layout
    print("Resulting Board Layout:")
    print_board(new_state)
    
    pieces_after = np.sum(new_state[game.KING])
    
    #Check if a piece was captured (count should decrease by one)
    print(f"Game pieces before: {pieces_before}, after: {pieces_after}")
    assert pieces_after == pieces_before - 2, "A game piece should have been captured!"

#testing shield wall captures
def test_shield_wall_captures(game, initial_layout):
     
    #shield wall against a hostile tile
    initial_layout[game.BLACK, 1, 1] = 1
    initial_layout[game.BLACK, 1, 2] = 1
    initial_layout[game.BLACK, 0, 4] = 1
    
    initial_layout[game.WHITE, 0, 1] = 1
    initial_layout[game.WHITE, 0, 2] = 1
    
    #shield wall against a the edge
    initial_layout[game.BLACK, 10, 4] = 1
    initial_layout[game.BLACK, 10, 5] = 1
    initial_layout[game.BLACK, 10, 6] = 1
    
    initial_layout[game.WHITE, 10, 2] = 1
    initial_layout[game.WHITE, 9, 4] = 1
    initial_layout[game.WHITE, 9, 5] = 1
    initial_layout[game.WHITE, 9, 6] = 1
    initial_layout[game.WHITE, 10, 7] = 1
    
    #Print initial board layout
    print("Initial Board Layout:")
    print_board(initial_layout)
    
    pieces_before = np.sum(initial_layout[game.WHITE]) + np.sum(initial_layout[game.BLACK])
    
    #capture two white against hostile square
    move = ((0, 4), (0, 3)) 
    new_state = game.make_move(initial_layout, move)
    
    #move to capture against edge
    move = ((10, 2), (10, 3)) 
    new_state = game.make_move(new_state, move)
    
    #Print resulting board layout
    print("Resulting Board Layout:")
    print_board(new_state)
    
    pieces_after = np.sum(new_state[game.WHITE]) + np.sum(new_state[game.BLACK])
    
    #Check if a piece was captured (count should decrease by one)
    print(f"Game pieces before: {pieces_before}, after: {pieces_after}")
    assert pieces_after == pieces_before - 5, "A game piece should have been captured!"

#Testing for an issue where pieces are being taken when attackers are 45* instead of 90 on each side
def test_offset_capture(game, initial_layout):
   
    initial_layout[game.WHITE, 4, 4] = 1 
    initial_layout[game.BLACK, 4, 2] = 1 
    initial_layout[game.BLACK, 3, 4] = 1  
    
    #Print initial board layout
    print("Initial Board Layout:")
    print_board(initial_layout)
    
    # Count Black pieces before the move
    pieces_before = np.sum(initial_layout[game.WHITE]) + np.sum(initial_layout[game.BLACK])
    
    #Make the move to capture the black piece
    move = ((4, 2), (4, 3))  # White moves from a7 to b7 to capture black
    new_state = game.make_move(initial_layout, move)
    
    # Print resulting board layout
    print("Resulting Board Layout:")
    print_board(new_state)
    
    # Count pieces after the move
    pieces_after = np.sum(new_state[game.WHITE]) + np.sum(new_state[game.BLACK])

    # Check if a piece was captured (count should decrease by two)
    print(f"Pieces before: {pieces_before}, after: {pieces_after}")
    assert pieces_after == pieces_before - 0, "A piece should have been captured!"

def test_surrounded_king(game, initial_layout):
    #Surounding the king
    initial_layout[game.BLACK, 3, 3] = 1
    initial_layout[game.BLACK, 3, 4] = 1
    initial_layout[game.BLACK, 3, 5] = 1
    initial_layout[game.BLACK, 3, 6] = 1
    initial_layout[game.BLACK, 3, 7] = 1
    
    initial_layout[game.BLACK, 4, 8] = 1
    initial_layout[game.BLACK, 5, 8] = 1
    initial_layout[game.BLACK, 6, 8] = 1
    
    initial_layout[game.BLACK, 4, 2] = 1
    initial_layout[game.BLACK, 5, 2] = 1
    initial_layout[game.BLACK, 6, 2] = 1
    
    initial_layout[game.BLACK, 7, 3] = 1
    initial_layout[game.BLACK, 7, 4] = 1
    initial_layout[game.BLACK, 7, 5] = 1
    initial_layout[game.BLACK, 7, 6] = 1
    initial_layout[game.BLACK, 7, 7] = 1
    
    initial_layout[game.WHITE, 4, 5] = 1
    initial_layout[game.WHITE, 6, 5] = 1
    initial_layout[game.WHITE, 5, 4] = 1
    initial_layout[game.WHITE, 5, 6] = 1
    
    initial_layout[game.KING, 5, 5] = 1
    
    #Print initial board layout
    print("Initial Board Layout:")
    print_board(initial_layout)
    
    initial_layout[game.PLAYER] = np.ones_like(initial_layout[game.PLAYER]) #force Black's turn
    
    # Check if game has ended by encirclement
    game_result = game.get_game_ended(initial_layout)
    
    # Print result
    print(f"Game result: {game_result}")
    print("1 means black (attackers) win, 0 means game continues")
    
    # Assert that black has won by encirclement
    assert game_result == 1, "Game should end with attackers (black) winning by encirclement!"


if __name__ == '__main__':
    pytest.main()