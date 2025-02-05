import numpy as np
import glob

# Mapping piece types to board channels
BLACK, WHITE, KING, SPECIAL, HISTORY, PLAYER = 0, 1, 2, 3, 4, 5

def algebraic_to_index(move):
    """Converts algebraic notation (e.g., "a1") to board indices (row, col)"""
    col = ord(move[0].lower()) - ord('a')  # Convert letter to index (a=0, ..., k=10)
    row = int(move[1:]) - 1        # Convert number to 0-based index (1-11 → 0-10)
    # Flip the row index since our board representation has (0,0) at the top
    row = 10 - row
    return row, col

def initialize_board():
    """Initializes a Hnefatafl board (11x11) with the correct starting position"""
    board = np.zeros((6, 11, 11), dtype=np.float32)
    
    # Create the initial layout - note that (0,0) is top-left in array representation
    initial_layout = np.array([
        # 11th rank (top)
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        # 10th rank
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # 9th rank
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # 8th rank
        [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
        # 7th rank
        [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
        # 6th rank (middle)
        [1, 1, 0, 2, 2, 3, 2, 2, 0, 1, 1],
        # 5th rank
        [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
        # 4th rank
        [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
        # 3rd rank
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        # 2nd rank
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        # 1st rank (bottom)
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    ])
    
    # Convert the single-channel representation to our multi-channel format
    board[BLACK] = (initial_layout == 1).astype(np.float32)  # Black pieces
    board[WHITE] = (initial_layout == 2).astype(np.float32)  # White pieces
    board[KING] = (initial_layout == 3).astype(np.float32)   # King
    
    # Mark special squares (corners and throne)
    corners = [(0, 0), (0, 10), (10, 0), (10, 10)]
    for r, c in corners:
        board[SPECIAL, r, c] = 1  # Corner squares
    board[SPECIAL, 5, 5] = 1      # Throne square (middle)
    
    # Initialize player channel (0 for black's turn, 1 for white's turn)
    board[PLAYER] = 0  # Black moves first
    
    return board

# [Rest of the code remains the same...]

# def print_board(board):
#     """
#     Prints the board using ASCII characters with coordinate labels.
#     Row numbers descend from 11 (top) to 1 (bottom).
#     Column letters (a-k) are shown at the bottom of the board.
#     """
#     display_board = np.zeros((11, 11), dtype=str)
#     display_board[:] = '.'
    
#     # Place pieces on display board in order of precedence
#     display_board[board[SPECIAL] == 1] = '#'    # Special squares first (corners and throne)
#     display_board[board[BLACK] == 1] = 'B'      # Black pieces next
#     display_board[board[WHITE] == 1] = 'W'      # White pieces next
#     display_board[board[KING] == 1] = 'K'       # King last (takes precedence)
    
#     # Print top border
#     print("  " + "-" * 23)                     
    
#     # Print each row with descending numbers (11 to 1)
#     for i, row in enumerate(display_board):
#         row_num = 11 - i                        # Convert row index to descending numbers
#         print(f"{row_num:2d}|{' '.join(row)}|") # Print row number and pieces
        
#     # Print bottom border
#     print("  " + "-" * 23)                     
    
#     # Print column letters at the bottom
#     print("   a b c d e f g h i j k")          
#     print()

def encode_move(start, end):
    """Encodes move as two heatmaps: (start position, end position)"""
    start_map = np.zeros((11, 11))
    end_map = np.zeros((11, 11))
    
    r1, c1 = start
    r2, c2 = end

    start_map[r1, c1] = 1
    end_map[r2, c2] = 1

    return start_map, end_map

def process_move(move_str):
    """Process a single move string, handling various formats and special cases"""
    if any(keyword in move_str.lower() for keyword in ['resigned', 'timeout', 'draw']):
        return None, None
    
    move_str = move_str.split('.')[-1].strip()  # Remove move numbers
    move_str = move_str.split('x')[0].strip()   # Remove capture markers
    
    if '-' not in move_str:
        return None, None
        
    try:
        start_str, end_str = move_str.split('-')
        start = algebraic_to_index(start_str)
        end = algebraic_to_index(end_str)
        return start, end
    except (ValueError, IndexError):
        return None, None

def process_game(game_string):
    """Process a single game and return training samples with game context"""
    parts = game_string.split(',')
    if len(parts) < 3:
        return []
        
    winner = parts[0].strip().lower()
    game_id = parts[1].strip()  # Get the game ID ID THIS IS GRABBING THE TOTAL MOVE COUNT BY MISTAIKE
    moves_list = parts[2].strip().split(' ')
    
    # Convert winner to numerical label
    if any(keyword in winner for keyword in ['resigned', 'timeout', 'draw']):
        winner_label = 0  # Draw or special case
    else:
        winner_label = 1 if "white" in winner else -1  # White wins: 1, Black wins: -1
    
    board = initialize_board()
    training_data = []
    move_number = 0  # Track move number within game

    print(f"\nProcessing Game {game_id}")
   #print("Initial Board:")
    #print_board(board)

    for move in moves_list:
        start, end = process_move(move)
        if start is None or end is None:
            continue
            
        print(f"Move {move_number + 1}: {move} ({start} -> {end})")
        
        start_map, end_map = encode_move(start, end)
        
        # Store more context with each training sample
        sample = {
            'board_state': board.copy(),
            'start_pos': start_map,
            'end_pos': end_map,
            'winner': winner_label,
            'game_id': game_id,
            'move_number': move_number,
            'player_turn': 'white' if board[PLAYER].sum() > 0 else 'black',
            'move_text': move
        }
        
        training_data.append(sample)

        # Apply move to board
        piece_type = None
        for channel in [BLACK, WHITE, KING]:
            if board[channel, start[0], start[1]] == 1:
                piece_type = channel
                break
                
        if piece_type is not None:
            board[piece_type, start[0], start[1]] = 0
            board[piece_type, end[0], end[1]] = 1

        # Switch player
        board[PLAYER] = 1 - board[PLAYER]
        move_number += 1

        #print_board(board)

    return training_data

def process_all_games():
    """Process all games and save with enhanced metadata"""
    all_data = []
    game_summaries = {}
    
    print("Starting game processing...")
    print("Looking for game files in current directory...")
    
    # Get list of matching files first
    game_files = glob.glob("game_moves_mini.csv")
    
    if not game_files:
        print("❌ No game_moves.csv file found in current directory!")
        print("Please ensure the file exists and is in the correct location.")
        return
        
    print(f"Found {len(game_files)} file(s): {game_files}")
    
    try:
        for file in game_files:
            print(f"\nReading file: {file}")
            with open(file, "r") as f:
                lines = f.readlines()
                print(f"Found {len(lines)} games to process")
                
                for i, line in enumerate(lines, 1):
                    print(f"\n{'='*50}")
                    print(f"Processing game {i} of {len(lines)}") #this is giving the total amount of moves. not the game number
                    print(f"Game data: {line[:100]}...")  # Show start of game data
                    
                    try:
                        game_data = process_game(line.strip())
                        
                        if game_data:
                            game_id = game_data[0]['game_id']
                            game_summaries[game_id] = {
                                'total_moves': len(game_data),
                                'winner': game_data[0]['winner'],
                                'final_position': game_data[-1]['board_state']
                            }
                            print(f"✅ Successfully processed game {game_id} with {len(game_data)} moves")
                            print(f"Winner: {'White' if game_data[0]['winner'] == 1 else 'Black' if game_data[0]['winner'] == -1 else 'Draw'}")
                        
                        all_data.extend(game_data)
                    except Exception as e:
                        print(f"❌ Error processing game {i}: {str(e)}")
                        print(f"Problematic line: {line.strip()}")
                        continue

        if not all_data:
            print("\n❌ No valid games were processed!")
            return

        print("\nPreparing to save processed data...")
        
        # Convert to numpy arrays
        X_boards = np.array([d['board_state'] for d in all_data])
        y_start = np.array([d['start_pos'] for d in all_data])
        y_end = np.array([d['end_pos'] for d in all_data])
        y_winner = np.array([d['winner'] for d in all_data])
        
        # Save dataset with metadata
        output_file = "hnefatafl_dataset.npz"
        np.savez(output_file,
                X_boards=X_boards, 
                y_start=y_start, 
                y_end=y_end, 
                y_winner=y_winner,
                metadata={
                    'game_ids': [d['game_id'] for d in all_data],
                    'move_numbers': [d['move_number'] for d in all_data],
                    'player_turns': [d['player_turn'] for d in all_data],
                    'move_texts': [d['move_text'] for d in all_data]
                })
        
        # Save game summaries
        summary_file = "game_summaries.npy"
        np.save(summary_file, game_summaries)
        
        print("\n✅ Processing complete!")
        print(f"Processed {len(game_summaries)} games with {len(all_data)} total moves")
        print(f"Data saved to {output_file}")
        print(f"Game summaries saved to {summary_file}")
        
    except Exception as e:
        print(f"\n❌ Fatal error during processing: {str(e)}")
        import traceback
        print(traceback.format_exc())

# Add a main block that provides more information about usage
if __name__ == "__main__":
    print("Hnefatafl Game Processor")
    print("=" * 50)
    print("This program processes Hnefatafl game records and converts them to training data.")
    print("Expected input: game_moves.csv file in the current directory")
    print("Format: winner,game_id,moves")
    print("Example: white won,game123,1. f6-f4 d4-d6 2. f4-f3 ...")
    print("=" * 50)
    print()
    
    process_all_games()