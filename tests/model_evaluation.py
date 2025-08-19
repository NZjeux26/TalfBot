import os
import sys
import torch
import numpy as np
import json
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
import time

# Add your project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "Data_Conversion_and_Validation"))
sys.path.append(os.path.join(project_root, "Training"))
sys.path.append(os.path.join(project_root, "Self_Training"))

from model import PolicyValueNetwork
from MCTSTwo import MCTS
from Move_Generation import HnefataflGame
from utils import get_device, Timer
class ModelEvaluator:
    def __init__(self, device=None):
        self.device = get_device()
        self.evaluation_history = []
        
    def load_model(self, model_path):
        """Load a model from file"""
        model = PolicyValueNetwork().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def get_policy_value_fn(self, model):
        """Create policy value function for a model"""
        def policy_value_fn(state):
            with torch.no_grad():
                if len(state.shape) == 3 and state.shape[0] == 6:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                else:
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    if len(state_tensor.shape) == 3:
                        state_tensor = state_tensor.unsqueeze(0)
                
                start_probs, end_probs, value = model(state_tensor)
                start_probs = start_probs.squeeze(0).cpu().numpy()
                end_probs = end_probs.squeeze(0).cpu().numpy()
                value = value.squeeze().cpu().item()
                
                return start_probs, end_probs, value
        return policy_value_fn
    
    def play_game(self, model1_fn, model2_fn, mcts_sims=None, verbose=False):
        """
        Play a single game between two models
        Returns: 1 if model1 wins, -1 if model2 wins, 0 if draw
        """
        # Initialize games for both models
        game1 = HnefataflGame(model1_fn)
        game2 = HnefataflGame(model2_fn)
        
        c_puct = testing_args['c_puct']
        
        # Initialize MCTS for both players
        mcts1 = MCTS(game1, n_simulations=mcts_sims, c_puct=c_puct)
        mcts2 = MCTS(game2, n_simulations=mcts_sims, c_puct=c_puct)
        
        state = game1.get_initial_state()
        move_count = 0
        max_moves = 200  # Prevent infinite games
        mcts_sims = testing_args['n_simulations']
        # Initialize move history for repetition detection
        move_history = []
        
        while move_count < max_moves:
            current_player = 'black' if state[game1.PLAYER].sum() == 0 else 'white'
            
            if verbose:
                print(f"Move {move_count + 1}: {current_player}'s turn")
            
            # Determine which model plays (model1 = black, model2 = white)
            if current_player == 'black':
                action_probs = mcts1.get_action_probs(state, temperature=testing_args['temperature'])
            else:
                action_probs = mcts2.get_action_probs(state, temperature=testing_args['temperature'])
            
            # Get valid moves for current position, if none break
            valid_moves = game1.get_valid_moves(state)
            if not valid_moves:
                print(f"No valid moves available for {current_player}!")
                game_result = 1 if current_player == 'black' else -1
                break
            
            # Select best move
            moves, probs = zip(*action_probs.items())
            moves = list(moves)
            probs = np.array(probs, dtype=float)
            probs /= probs.sum()  # ensure probabilities sum to 1

            idx = np.random.choice(len(moves), p=probs)
            move = moves[idx]
            
            # Make move
            state = game1.make_move(state, move)
            move_count += 1
            
            # Check for repetition (after the move is made)
            if game1.check_move_repetition(move_history, move):
                game1.print_state(state)
                print("\nGame Over! Black wins by perpetual repetition!")
                return 1 # changed from break
            
            # Check for game end
            result = game1.get_game_ended(state)
            if result != 0:
                if verbose:
                    winner = "Black (Model1)" if result == 1 else "White (Model2)"
                    print(f"Game over! {winner} wins in {move_count} moves")
                return result
        
        if verbose:
            print(f"Game reached maximum moves ({max_moves}), declaring draw")
        return 0  # Draw
    
    def tournament(self, model1_path, model2_path, num_games=None, mcts_sims=None):
        """
        Run a tournament between two models
        Each model plays as both black and white
        """
        print(f"\n=== Tournament: {os.path.basename(model1_path)} vs {os.path.basename(model2_path)} ===")
        print(f"Games: {num_games}, MCTS simulations: {mcts_sims}")
        
        num_games = testing_args['num_games']
        mcts_sims = testing_args['n_simulations']
        
        model1 = self.load_model(model1_path)
        model2 = self.load_model(model2_path)
        
        model1_fn = self.get_policy_value_fn(model1)
        model2_fn = self.get_policy_value_fn(model2)
        
        results = []
        model1_wins = 0
        model2_wins = 0
        draws = 0
        
        start_time = time.time()
        
        for game_num in range(num_games):
            print(f"Playing game {game_num + 1}/{num_games}...", end=" ")
            
            # Alternate who plays as black/white
            if game_num % 2 == 0:
                # Model1 as black, Model2 as white
                result = self.play_game(model1_fn, model2_fn, mcts_sims)
                if result == 1:
                    model1_wins += 1
                    print("Model1 wins (Black)")
                elif result == -1:
                    model2_wins += 1
                    print("Model2 wins (White)")
                else:
                    draws += 1
                    print("Draw")
            else:
                # Model2 as black, Model1 as white
                result = self.play_game(model2_fn, model1_fn, mcts_sims)
                if result == 1:
                    model2_wins += 1
                    print("Model2 wins (Black)")
                elif result == -1:
                    model1_wins += 1
                    print("Model1 wins (White)")
                else:
                    draws += 1
                    print("Draw")
            
            results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        total_decisive_games = model1_wins + model2_wins
        model1_winrate = (model1_wins / total_decisive_games * 100) if total_decisive_games > 0 else 0
        model2_winrate = (model2_wins / total_decisive_games * 100) if total_decisive_games > 0 else 0
        
        tournament_result = {
            'timestamp': datetime.now().isoformat(),
            'model1_path': model1_path,
            'model2_path': model2_path,
            'num_games': num_games,
            'mcts_sims': mcts_sims,
            'model1_wins': model1_wins,
            'model2_wins': model2_wins,
            'draws': draws,
            'model1_winrate': model1_winrate,
            'model2_winrate': model2_winrate,
            'elapsed_time': elapsed_time,
            'results': results
        }
        
        print(f"\n=== Tournament Results ===")
        print(f"Model1 wins: {model1_wins} ({model1_winrate:.1f}%)")
        print(f"Model2 wins: {model2_wins} ({model2_winrate:.1f}%)")
        print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
        print(f"Total time: {elapsed_time:.1f} seconds")
        
        return tournament_result
    
    def benchmark_against_baseline(self, new_model_path, baseline_model_path, num_games=None):
        """
        Test new model against baseline to measure improvement
        Returns True if new model is significantly better
        """
        num_games = testing_args['num_games']
        result = self.tournament(new_model_path, baseline_model_path, num_games)
        
        # Consider improvement if new model wins >55% of decisive games
        improvement_threshold = 55.0
        is_improved = result['model1_winrate'] > improvement_threshold
        
        print(f"\n=== Improvement Assessment ===")
        print(f"New model winrate: {result['model1_winrate']:.1f}%")
        print(f"Improvement threshold: {improvement_threshold}%")
        print(f"Model improved: {'YES' if is_improved else 'NO'}")
        
        return is_improved, result
    
    def evaluate_position_accuracy(self, model_path, test_positions=None):
        """
        Evaluate how accurately the model assesses known positions
        You can add specific test positions with known evaluations
        """
        model = self.load_model(model_path)
        policy_fn = self.get_policy_value_fn(model)
        game = HnefataflGame(policy_fn)
        
        if test_positions is None:
            # Create some standard test positions
            test_positions = []
            
            # Starting position
            initial_state = game.get_initial_state()
            test_positions.append(('initial_position', initial_state, 0.0))  # Should be roughly balanced
            
            # You can add more specific positions here
            # test_positions.append(('winning_position', state, expected_value))
        
        evaluations = []
        
        print(f"\n=== Position Evaluation Test ===")
        for pos_name, state, expected_value in test_positions:
            _, _, predicted_value = policy_fn(state)
            error = abs(predicted_value - expected_value)
            
            evaluations.append({
                'position': pos_name,
                'expected': expected_value,
                'predicted': predicted_value,
                'error': error
            })
            
            print(f"{pos_name}: Expected {expected_value:.3f}, Predicted {predicted_value:.3f}, Error {error:.3f}")
        
        avg_error = np.mean([e['error'] for e in evaluations])
        print(f"Average evaluation error: {avg_error:.3f}")
        
        return evaluations
    
    def save_evaluation_history(self, filename="evaluation_history.json"):
        """Save all evaluation results to file"""
        with open(filename, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        print(f"Evaluation history saved to {filename}")
    
    def plot_improvement_over_time(self, save_path="model_improvement.png"):
        """Plot model improvement over training iterations"""
        if not self.evaluation_history:
            print("No evaluation history to plot")
            return
        
        iterations = []
        winrates = []
        
        for result in self.evaluation_history:
            if 'iteration' in result:
                iterations.append(result['iteration'])
                winrates.append(result['model1_winrate'])
        
        if not iterations:
            print("No iteration data found in evaluation history")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, winrates, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Training Iteration')
        plt.ylabel('Winrate vs Baseline (%)')
        plt.title('Model Improvement Over Training')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='Baseline (50%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Improvement plot saved to {save_path}")
    
    def play_game_with_visualisation(self, model1_fn, model2_fn, mcts_sims=None, verbose=False, print_every_n_moves=10):
        """
        Play a single game with visual output for debugging
        """
        game1 = HnefataflGame(model1_fn)
        game2 = HnefataflGame(model2_fn)
        
        c_puct = testing_args['c_puct']
        
        mcts1 = MCTS(game1, n_simulations=mcts_sims, c_puct=c_puct)
        mcts2 = MCTS(game2, n_simulations=mcts_sims, c_puct=c_puct)
        
        state = game1.get_initial_state()
        move_count = 0
        max_moves = 200
        mcts_sims = testing_args['n_simulations']
        move_history = []
        
        print(f"\n=== GAME START ===")
        game1.print_state(state)
        
        while move_count < max_moves:
            current_player = 'black' if state[game1.PLAYER].sum() == 0 else 'white'
            
            # Get model evaluation
            _, _, value = model1_fn(state) if current_player == 'black' else model2_fn(state)
            
            if verbose or (move_count % print_every_n_moves == 0):
                print(f"\n--- Move {move_count + 1}: {current_player}'s turn ---")
                print(f"Model evaluation: {value:.3f}")
            
            if current_player == 'black':
                action_probs = mcts1.get_action_probs(state, temperature=testing_args['temperature'])
            else:
                action_probs = mcts2.get_action_probs(state, temperature=testing_args['temperature'])
            
            valid_moves = game1.get_valid_moves(state)
            if not valid_moves:
                print(f"No valid moves available for {current_player}!")
                return 1 if current_player == 'black' else -1
            
            # Sample move properly
            moves, probs = zip(*action_probs.items())
            moves_list = list(moves)
            probs = np.array(probs)
            probs = probs / np.sum(probs)
            chosen_idx = np.random.choice(len(moves_list), p=probs)
            move = moves_list[chosen_idx]
            
            # Convert move to algebraic notation for display
            start_alg = chr(ord('a') + move[0][1]) + str(11 - move[0][0])
            end_alg = chr(ord('a') + move[1][1]) + str(11 - move[1][0])
            
            if verbose or (move_count % print_every_n_moves == 0):
                print(f"{current_player} plays: {start_alg}-{end_alg}")
                print(f"Move probability: {probs[chosen_idx]:.3f}")
            
            # Make move
            state = game1.make_move(state, move)
            move_history.append(move)
            move_count += 1
            
            # Print board state every N moves
            if verbose or (move_count % print_every_n_moves == 0):
                game1.print_state(state)
            
            # Check for repetition
            if game1.check_move_repetition(move_history, move):
                game1.print_state(state)
                print("\nGame Over! Black wins by perpetual repetition!")
                return 1
            
            # Check for game end
            result = game1.get_game_ended(state)
            if result != 0:
                game1.print_state(state)
                winner = "Black (Model1)" if result == 1 else "White (Model2)"
                print(f"Game over! {winner} wins in {move_count} moves")
                return result
        
        print(f"\n=== GAME ENDED IN DRAW (MAX MOVES: {max_moves}) ===")
        game1.print_state(state)
        return 0
    
    def debug_tournament(self, model1_path, model2_path, num_games=1):
        """
        Run a tournament with full visualization for debugging
        """
        print(f"\n=== DEBUG TOURNAMENT ===")
        print(f"Model 1: {os.path.basename(model1_path)}")
        print(f"Model 2: {os.path.basename(model2_path)}")
        
        model1 = self.load_model(model1_path)
        model2 = self.load_model(model2_path)
        
        model1_fn = self.get_policy_value_fn(model1)
        model2_fn = self.get_policy_value_fn(model2)
        
        for game_num in range(num_games):
            print(f"\n" + "="*60)
            print(f"GAME {game_num + 1}/{num_games}")
            if game_num % 2 == 0:
                print("Model1 (Black) vs Model2 (White)")
                result = self.play_game_with_visualisation(model1_fn, model2_fn, 
                                                        mcts_sims=testing_args['n_simulations'], 
                                                        verbose=False, print_every_n_moves=20)
            else:
                print("Model2 (Black) vs Model1 (White)")
                result = self.play_game_with_visualisation(model2_fn, model1_fn, 
                                                        mcts_sims=testing_args['n_simulations'], 
                                                        verbose=False, print_every_n_moves=20)
            
            if result == 0:
                print(f"Game {game_num + 1} ended in DRAW")
            else:
                winner = "Model1" if (result == 1 and game_num % 2 == 0) or (result == -1 and game_num % 2 == 1) else "Model2"
                print(f"Game {game_num + 1} won by {winner}")

def comprehensive_evaluation(new_model_path, baseline_model_path, iteration=None):
    """
    Run a comprehensive evaluation of a new model
    """
    evaluator = ModelEvaluator()
    
    print(f"=" * 60)
    print(f"COMPREHENSIVE MODEL EVALUATION")
    print(f"New model: {os.path.basename(new_model_path)}")
    print(f"Baseline: {os.path.basename(baseline_model_path)}")
    print(f"=" * 60)
    
    # 1. Quick tournament (10 games)
    print("\n1. Quick Tournament (10 games)")
    quick_result = evaluator.tournament(new_model_path, baseline_model_path, num_games=testing_args['num_games'], mcts_sims=testing_args['n_simulations'])
    
    # 2. Detailed tournament (30 games) if quick test shows promise
    if quick_result['model1_winrate'] > 45:  # If showing some promise
        print("\n2. Detailed Tournament (30 games)")
        detailed_result = evaluator.tournament(new_model_path, baseline_model_path, num_games=testing_args['num_games'], mcts_sims=testing_args['n_simulations'])
    else:
        print("\n2. Skipping detailed tournament (model performing poorly)")
        detailed_result = quick_result
    
    # 3. Position evaluation test
    print("\n3. Position Evaluation Test")
    position_evals = evaluator.evaluate_position_accuracy(new_model_path)
    
    # 4. Store results
    evaluation_summary = {
        'iteration': iteration,
        'timestamp': datetime.now().isoformat(),
        'new_model_path': new_model_path,
        'baseline_model_path': baseline_model_path,
        'quick_tournament': quick_result,
        'detailed_tournament': detailed_result,
        'position_evaluations': position_evals,
        'model1_winrate': detailed_result['model1_winrate'],
        'is_improved': detailed_result['model1_winrate'] > 55
    }
    
    evaluator.evaluation_history.append(evaluation_summary)
    evaluator.save_evaluation_history()
    
    # 5. Summary
    print(f"\n" + "=" * 60)
    print(f"EVALUATION SUMMARY")
    print(f"Winrate: {detailed_result['model1_winrate']:.1f}%")
    print(f"Improved: {'YES' if evaluation_summary['is_improved'] else 'NO'}")
    print(f"Games played: {detailed_result['num_games']}")
    print(f"=" * 60)
    
    return evaluation_summary

            
if __name__ == "__main__": #Need to mod this to pick which model to compare and comprehensive or quick evaluation``
    
    #Paths to models to be tested
    main_model = "hnefatafl_policy_value_model.pth"
    model_to_test = "models/hnefatafl_selfplay_iter_final_20250802_232744.pth"
    
    testing_args = {
        'c_puct': 1.2,              # MCTS exploration constant
        'n_simulations': 100,       # MCTS simulations per move must be > 0
        'temperature': 1.0,         # Move selection temperature
        'num_games': 2,             # Games per iteration
    }
    
    # Example usage
    print("Model Evaluation Framework")
    
    # Example comparison (uncomment and modify paths as needed)
    # comprehensive_evaluation(
    #     model_to_test,
    #     main_model,
    #     iteration=5
    # )
    evaluator = ModelEvaluator()
    
    # Run debug tournament with visualization
    evaluator.debug_tournament(
        model_to_test,
        main_model,
        num_games=1  # Just one game with full output
    )