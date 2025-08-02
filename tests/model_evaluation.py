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

from model import PolicyValueNetwork
from MCTSTwo import MCTS
from Move_Generation import HnefataflGame
from utils import get_device
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
    
    def play_game(self, model1_fn, model2_fn, mcts_sims=400, verbose=False):
        """
        Play a single game between two models
        Returns: 1 if model1 wins, -1 if model2 wins, 0 if draw
        """
        # Initialize games for both models
        game1 = HnefataflGame(model1_fn)
        game2 = HnefataflGame(model2_fn)
        
        # Initialize MCTS for both players
        mcts1 = MCTS(game1, n_simulations=mcts_sims, c_puct=1.0)
        mcts2 = MCTS(game2, n_simulations=mcts_sims, c_puct=1.0)
        
        state = game1.get_initial_state()
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        # Initialize move history for repetition detection
        move_history = []
        
        while move_count < max_moves:
            current_player = 'black' if state[game1.PLAYER].sum() == 0 else 'white'
            
            if verbose:
                print(f"Move {move_count + 1}: {current_player}'s turn")
            
            # Determine which model plays (model1 = black, model2 = white)
            if current_player == 'black':
                action_probs = mcts1.get_action_probs(state, temperature=0.1)
            else:
                action_probs = mcts2.get_action_probs(state, temperature=0.1)
            
            # Get valid moves for current position, if none break
            valid_moves = self.game.get_valid_moves(state)
            if not valid_moves:
                print(f"No valid moves available for {current_player}!")
                game_result = 1 if current_player == 'black' else -1
                break
            
            # Select best move
            moves, probs = zip(*action_probs.items())
            move = moves[np.argmax(probs)]
            
            # Make move
            state = game1.make_move(state, move)
            move_count += 1
            
            # Check for repetition (after the move is made)
            if self.game.check_move_repetition(move_history, move):
                self.game.print_state(state)
                print("\nGame Over! Black wins by perpetual repetition!")
                break
            
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
    
    def tournament(self, model1_path, model2_path, num_games=20, mcts_sims=400):
        """
        Run a tournament between two models
        Each model plays as both black and white
        """
        print(f"\n=== Tournament: {os.path.basename(model1_path)} vs {os.path.basename(model2_path)} ===")
        print(f"Games: {num_games}, MCTS simulations: {mcts_sims}")
        
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
    
    def benchmark_against_baseline(self, new_model_path, baseline_model_path, num_games=30):
        """
        Test new model against baseline to measure improvement
        Returns True if new model is significantly better
        """
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
    quick_result = evaluator.tournament(new_model_path, baseline_model_path, num_games=10, mcts_sims=200)
    
    # 2. Detailed tournament (30 games) if quick test shows promise
    if quick_result['model1_winrate'] > 45:  # If showing some promise
        print("\n2. Detailed Tournament (30 games)")
        detailed_result = evaluator.tournament(new_model_path, baseline_model_path, num_games=30, mcts_sims=400)
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

if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Framework")
    print("Usage examples:")
    print("1. Compare two models:")
    print("   python model_evaluation.py")
    print("2. Use in training loop to evaluate each iteration")
    
    # Example comparison (uncomment and modify paths as needed)
    # comprehensive_evaluation(
    #     "models/hnefatafl_selfplay_iter_5.pth",
    #     "hnefatafl_policy_value_model.pth",
    #     iteration=5
    # )