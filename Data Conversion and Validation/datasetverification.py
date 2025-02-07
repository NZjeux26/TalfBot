import numpy as np

# Load dataset
data = np.load("hnefatafl_dataset.npz")

# Extract arrays
X_boards = data["X_boards"]  # (num_samples, 6, 11, 11)
y_start = data["y_start"]    # (num_samples, 11, 11)
y_end = data["y_end"]        # (num_samples, 11, 11)
y_winner = data["y_winner"]  # (num_samples,)

# Print dataset shape
print(f"Board states: {X_boards.shape}")
print(f"Move start maps: {y_start.shape}")
print(f"Move end maps: {y_end.shape}")
print(f"Winner labels: {y_winner.shape}")

# Check unique winner labels
print(f"Unique winner labels: {np.unique(y_winner)}")

def print_board_from_tensor(board):
    """Prints a board state from the dataset"""
    symbols = {0: '.', 1: 'B', 2: 'W', 3: 'K', 4: '#'}
    display_board = [['.' for _ in range(11)] for _ in range(11)]

    for r in range(11):
        for c in range(11):
            if board[0, r, c] == 1:
                display_board[r][c] = 'B'  # Black piece
            elif board[1, r, c] == 1:
                display_board[r][c] = 'W'  # White piece
            elif board[2, r, c] == 1:
                display_board[r][c] = 'K'  # King
            elif board[3, r, c] == 1:
                display_board[r][c] = '#'  # Special square

    print("\n".join([" ".join(row) for row in display_board]))
    print("\n")

# Print first 5 board states
for i in range(5):
    print(f"Board State {i+1}:")
    print_board_from_tensor(X_boards[i])

def visualise_move(y_start, y_end):
    """Visualises a move's start and end heatmaps"""
    start_positions = np.argwhere(y_start == 1)
    end_positions = np.argwhere(y_end == 1)

    print(f"Start Positions: {start_positions}")
    print(f"End Positions: {end_positions}")

# Check first 5 moves
for i in range(5):
    print(f"Move {i+1}:")
    visualise_move(y_start[i], y_end[i])

# Count occurrences of White (1) and Black (-1) wins
white_wins = np.sum(y_winner == 1)
black_wins = np.sum(y_winner == -1)

print(f"White Wins: {white_wins}")
print(f"Black Wins: {black_wins}")

