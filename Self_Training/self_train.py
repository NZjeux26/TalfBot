import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from datetime import datetime
from copy import deepcopy

# Add your project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root, "Data_Conversion_and_Validation"))
sys.path.append(os.path.join(project_root, "Training"))
sys.path.append(os.path.join(project_root, "tests"))

# Import your components
from model import PolicyValueNetwork
from MCTSTwo import MCTS
from Move_Generation import HnefataflGame
from utils import get_device
# Import evaluation framework
try:
    from model_evaluation import comprehensive_evaluation
    EVALUATION_AVAILABLE = False #Forcing false for now.
except ImportError:
    EVALUATION_AVAILABLE = False
    print("Warning: model_evaluation.py not found. Evaluation features disabled.")

class SelfPlay:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.mcts = MCTS(game, 
                        c_puct=args['c_puct'],
                        n_simulations=args['n_simulations'])
        self.temperature = args['temperature']
        self.num_games = args['num_games']
        
    def execute_episode(self):
        """Play one episode and return training data"""
        train_examples = []
        state = self.game.get_initial_state()
        temperature = self.temperature
        
        while True:
            # Get MCTS action probabilities
            action_probs = self.mcts.get_action_probs(state, temperature)
            
            # Store the training example
            train_examples.append({
                'state': state.copy(),
                'action_probs': action_probs,
                'player': state[self.game.PLAYER].sum()  # 0 for black, 1 for white
            })
            
            # Select move
            moves, probs = zip(*action_probs.items())
            move = moves[np.random.choice(len(moves), p=probs)]
            
            # Execute move
            state = self.game.make_move(state, move)
            
            # Check if game is ended
            game_result = self.game.get_game_ended(state)
            if game_result != 0:
                # Add game result to all examples
                for example in train_examples:
                    player = example['player']
                    if (game_result == 1 and player == 0) or (game_result == -1 and player == 1):
                        example['value'] = 1
                    else:
                        example['value'] = -1
                break
            
            # Gradually reduce temperature
            if len(train_examples) == 15:  # After 10 moves
                temperature = 1.5
            elif len(train_examples) == 30:  # After 20 moves
                temperature = 0.9
            elif len(train_examples) < 45:    # Mid-game
                temperature = 0.7
            else:                            # End-game
                temperature = 0.4
        return train_examples
    
    def self_play(self):
        """Execute multiple self-play episodes and return training data"""
        all_examples = []
        
        for i in range(self.num_games):
            print(f"Playing game {i+1}/{self.num_games}")
            examples = self.execute_episode()
            all_examples.extend(examples)
            
        return all_examples

def train_network(model, examples, args):
    """Train the network using self-play examples"""
    optimiser = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    batch_size = args['batch_size']
    
    # Convert examples to tensors
    states = torch.FloatTensor([ex['state'] for ex in examples])
    action_probs = torch.FloatTensor([list(ex['action_probs'].values()) for ex in examples])
    values = torch.FloatTensor([ex['value'] for ex in examples])
    
    dataset = torch.utils.data.TensorDataset(states, action_probs, values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    
    for batch_idx, (state, target_probs, target_value) in enumerate(dataloader):
        optimiser.zero_grad()
        
        # Forward pass
        start_probs, end_probs, value = model(state)
        
        # Calculate loss
        policy_loss = -(target_probs * torch.log(start_probs)).sum(dim=1).mean()
        policy_loss += -(target_probs * torch.log(end_probs)).sum(dim=1).mean()
        value_loss = ((value - target_value) ** 2).mean()
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        optimiser.step()
    
    model.eval()
    
class SelfPlayTrainer:
    def __init__(self, model_path=None, device=None):
        self.device = get_device() #turns out all other versions are putting the device call in the method call.
        print(f"Using device: {self.device}")
        self.model = PolicyValueNetwork().to(self.device)
        self.baseline_model_path = model_path   # Keep track of baseline for evaluation
        
        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            # Load to CPU first, then move to target device
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(device)  # Move model to target device
        else:
            print("Starting with random model weights")
        
        # Initialize game with model's policy function
        self.game = HnefataflGame(self.get_policy_value)
        
        # Training examples storage
        self.training_examples = []
        self.max_examples = 400000  # Keep last N examples was 200k
        
    def get_policy_value(self, state):
        """Get policy and value predictions from the model - wrapper for HnefataflGame"""
        with torch.no_grad():
            if len(state.shape) == 3 and state.shape[0] == 6:  # Expected format [6, 11, 11]
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                # Handle other formats if needed
                state_tensor = torch.FloatTensor(state).to(self.device)
                if len(state_tensor.shape) == 3:
                    state_tensor = state_tensor.unsqueeze(0)
            
            start_probs, end_probs, value = self.model(state_tensor)
            
            # Ensure proper tensor handling
            start_probs = start_probs.squeeze(0).cpu().numpy()
            end_probs = end_probs.squeeze(0).cpu().numpy()
            value = value.squeeze().cpu().item()
            
            return start_probs, end_probs, value
    
    def self_play_iteration(self, iteration, args):
        """Execute one iteration of self-play and training"""
        print(f"\n=== Self-Play Iteration {iteration} ===")
        
        # Execute self-play games
        print(f"Generating {args['num_games']} self-play games...")
        self_play = SelfPlay(self.game, self.model, args)
        new_examples = self_play.self_play()
        
        # Add new examples to training set
        self.training_examples.extend(new_examples)
        
        # Keep only the most recent examples
        if len(self.training_examples) > self.max_examples:
            self.training_examples = self.training_examples[-self.max_examples:]
        
        print(f"Total training examples: {len(self.training_examples)}")
        
        # Train the network
        if len(self.training_examples) > args['min_examples']:
            print("Training neural network...")
            self._train_network_updated(self.training_examples, args)
            print("Training completed")
        else:
            print(f"Need {args['min_examples']} examples to start training, have {len(self.training_examples)}")
    
    def _train_network_updated(self, examples, args):
        """Updated training function that handles the specific format"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args['learning_rate'])
        batch_size = args['batch_size']
        epochs = args['epochs']
        
        # Convert examples to the right format
        states = []
        start_target_probs = []
        end_target_probs = []
        values = []
        
        for ex in examples:
            states.append(ex['state'])
            
            # Convert action_probs dict to start/end probability arrays
            action_probs = ex['action_probs']
            start_probs = np.zeros(121)  # 11x11 = 121
            end_probs = np.zeros(121)
            
            for (start, end), prob in action_probs.items():
                start_idx = start[0] * 11 + start[1]
                end_idx = end[0] * 11 + end[1]
                start_probs[start_idx] += prob
                end_probs[end_idx] += prob
            
            # Normalize probabilities
            if start_probs.sum() > 0:
                start_probs = start_probs / start_probs.sum()
            if end_probs.sum() > 0:
                end_probs = end_probs / end_probs.sum()
                
            start_target_probs.append(start_probs)
            end_target_probs.append(end_probs)
            values.append(ex['value'])
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        start_target_probs = torch.FloatTensor(np.array(start_target_probs)).to(self.device)
        end_target_probs = torch.FloatTensor(np.array(end_target_probs)).to(self.device)
        values = torch.FloatTensor(np.array(values)).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(states, start_target_probs, end_target_probs, values)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            batches = 0
            
            for batch_idx, (state_batch, start_target, end_target, value_target) in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Forward pass
                start_pred, end_pred, value_pred = self.model(state_batch)
                
                # Calculate losses
                start_policy_loss = F.cross_entropy(start_pred, start_target)
                end_policy_loss = F.cross_entropy(end_pred, end_target)
                value_loss = F.mse_loss(value_pred.squeeze(), value_target)
                
                # Combined loss
                loss = start_policy_loss + end_policy_loss + value_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batches += 1
            
            if epoch == 0 or (epoch + 1) % 2 == 0:
                avg_loss = total_loss / batches if batches > 0 else 0
                print(f"  Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        self.model.eval()
    
    def save_model(self, iteration, save_dir="models"):
        """Save the current model"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(save_dir, f"hnefatafl_selfplay_iter_{iteration}_{timestamp}.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        return model_path
    
    def save_training_examples(self, iteration, save_dir="training_data"):
        """Save training examples for backup/analysis"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        examples_path = os.path.join(save_dir, f"training_examples_iter_{iteration}_{timestamp}.pkl")
        with open(examples_path, 'wb') as f:
            pickle.dump(self.training_examples, f)
        print(f"Training examples saved to {examples_path}")
    
    def train(self, num_iterations=None, args=None):
        """Main training loop"""
        if args is None:
            raise ValueError("Training arguments must be provided")
        
        print("Starting self-play training...")
        print(f"Configuration: {args}")
        
        for iteration in range(1, num_iterations + 1):
            try:
                # Execute self-play and training
                self.self_play_iteration(iteration, args)
                
                # Save model every few iterations
                if iteration % 2 == 0:
                    model_path = self.save_model(iteration)
                    
                    # Evaluate model improvement every 2 iterations if baseline exists
                    if EVALUATION_AVAILABLE and self.baseline_model_path and iteration >= 2:
                        print(f"\n--- Evaluating Model Improvement ---")
                        try:
                            eval_result = comprehensive_evaluation(
                                model_path, 
                                self.baseline_model_path, 
                                iteration=iteration
                            )
                            
                            # Update baseline if model improved significantly
                            if eval_result['is_improved'] and eval_result['model1_winrate'] > 60:
                                print(f"Model shows significant improvement! Updating baseline.")
                                self.baseline_model_path = model_path
                                
                        except Exception as e:
                            print(f"Evaluation failed: {e}")
                
                # Save training examples every 5 iterations
                if iteration % 5 == 0:
                    self.save_training_examples(iteration)
                
                print(f"Iteration {iteration} completed successfully")
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                # Save current state before continuing
                self.save_model(f"{iteration}_error")
                self.save_training_examples(f"{iteration}_error")
                raise e
        
        # Final save
        final_model_path = self.save_model("final")
        print(f"\nTraining completed! Final model saved to: {final_model_path}")

def main():
    # Training configuration
    training_args = {
        'c_puct': 1.0,              # MCTS exploration constant
        'n_simulations': 1,       # MCTS simulations per move
        'temperature': 1.0,         # Move selection temperature
        'num_games': 1,            # Games per iteration
        'learning_rate': 0.001,     # Neural network learning rate
        'batch_size': 32,           # Training batch size
        'min_examples': 1000,       # Minimum examples before training starts
        'epochs': 5,                 # Training epochs per iteration
        'num_iterations': 10
        
    }
    
    # Initialize trainer
    trainer = SelfPlayTrainer(
        model_path="hnefatafl_policy_value_model.pth"  # Your existing model
    )
    
    # Start training
    num_iterations = training_args['num_iterations']
    trainer.train(num_iterations, training_args)
    
    # Final comprehensive evaluation if evaluation framework is available
    if EVALUATION_AVAILABLE and trainer.baseline_model_path:
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        final_model_path = trainer.save_model("final_evaluation")
        comprehensive_evaluation(
            final_model_path,
            "hnefatafl_policy_value_model.pth",  # Original baseline
            iteration="final"
        )

if __name__ == "__main__":
    print("Hnefatafl Self-Play Training")
    print("=" * 50)
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nTraining session ended.")