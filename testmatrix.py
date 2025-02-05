import numpy as np
import re

# 🏰 Define the Copenhagen Hnefatafl 11×11 Starting Board
def get_initial_board():
    board = np.zeros((11, 11), dtype=int)

    # Defenders (White)
    defenders = [(5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (3, 5), (4, 5), (6, 5), (7, 5)]
    for x, y in defenders:
        board[x, y] = 2  # White defenders

    # King (White)
    board[5, 5] = 3

    # Attackers (Black)
    attackers = [
        (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 5), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0),
        (9, 5), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),
        (3, 10), (4, 10), (5, 10), (6, 10), (7, 10)
    ]
    for x, y in attackers:
        board[x, y] = 1  # Black attackers

    return board

# 📍 Convert algebraic notation (e.g., "k8-i8") to matrix indices
def notation_to_indices(move):
    column_map = {ch: i for i, ch in enumerate("abcdefghijk")}
    
    match = re.match(r"([a-k])(\d+)-([a-k])(\d+)", move)
    if match:
        col1, row1, col2, row2 = match.groups()
        return (int(row1) - 1, column_map[col1]), (int(row2) - 1, column_map[col2])
    return None, None

# 🔄 Apply a move to the board
def apply_move(board, move):
    if move.lower() in ["resigned", "timeout"]:
        return board  # No change to the board state
    start, end = notation_to_indices(move)
    if start and end:
        board[end] = board[start]  # Move piece
        board[start] = 0  # Clear old position
    return board

# 🔄 Process a game and generate board states
def process_game(moves_list):
    board = get_initial_board()
    board_states = []

    for move_pair in moves_list:
        moves = move_pair.split()

        # 🛠️ Ensure each move pair has both Black and White moves
        if len(moves) == 2:
            black_move, white_move = moves
            board = apply_move(board.copy(), black_move)
            board_states.append(board.copy())  # Store black's move state
            board = apply_move(board.copy(), white_move)
            board_states.append(board.copy())  # Store white's move state

        elif len(moves) == 1:
            # If only one move is present (game ended early), apply it
            board = apply_move(board.copy(), moves[0])
            board_states.append(board.copy())

    return np.array(board_states)  # Shape: (N, 11, 11)

# 📂 Example Usage
game_data = "1. k8-i8 d6-d9 2. a4-c4 g7-i7 3. h11-h9 g6-g10 4. e11-e10 resigned"
moves_list = game_data.split(" ")[1:]  # Extract moves
board_states = process_game(moves_list)

# ✅ Save to NumPy File
np.save("hnefatafl_board_states.npy", board_states)
print(f"✅ Saved {len(board_states)} board states to 'hnefatafl_board_states.npy'")

# ✅ Save to CSV (Optional)
np.savetxt("hnefatafl_board_states.csv", board_states.reshape(len(board_states), -1), delimiter=",")
print(f"✅ Also saved as CSV: 'hnefatafl_board_states.csv'")

# ✅ Display one board state as example
print("\nExample board state (after move 1):")
print(board_states[0])
