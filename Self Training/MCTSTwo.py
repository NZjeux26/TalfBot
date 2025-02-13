import numpy as np
from collections import defaultdict
import math
import torch
from copy import deepcopy

class MCTSNode:
    def __init__(self, game_state, prior=None, move=None, parent=None):
        self.game_state = game_state
        self.move = move  # Move that led to this state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior or 0
        self.is_expanded = False

class MCTS:
    def __init__(self, game, c_puct=1.0, n_simulations=800):
        self.game = game
        self.c_puct = c_puct
        self.n_simulations = n_simulations
    
    def search(self, root_state):
        root = MCTSNode(root_state)
        
        for _ in range(self.n_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.is_expanded:
                action, node = self._select_child(node)
                search_path.append(node)
            
            # Get game state at selected node
            game_state = node.game_state
            
            # Check if game is ended
            value = self.game.get_game_ended(game_state)
            
            if value == 0:
                # Expansion
                valid_moves = self.game.get_valid_moves(game_state)
                move_probs = self.game.get_move_probabilities(game_state, valid_moves)
                
                # Create children nodes
                for move, prob in move_probs.items():
                    next_state = self.game.make_move(game_state, move)
                    node.children[move] = MCTSNode(
                        next_state,
                        prior=prob,
                        move=move,
                        parent=node
                    )
                node.is_expanded = True
                
                # Get value from neural network
                _, _, value = self.game.get_policy_value_predictions(game_state)
            
            # Backpropagate
            self._backpropagate(search_path, value)
        
        return root
    
    def _select_child(self, node):
        """Select child using PUCT algorithm"""
        total_sqrt = math.sqrt(node.visit_count)
        
        def ucb_score(child):
            prior_score = self.c_puct * child.prior * total_sqrt / (1 + child.visit_count)
            if child.visit_count > 0:
                value_score = -child.value_sum / child.visit_count
            else:
                value_score = 0
            return value_score + prior_score
        
        move, child = max(node.children.items(), key=lambda x: ucb_score(x[1]))
        return move, child
    
    def _backpropagate(self, search_path, value):
        """Update statistics of all nodes in search path"""
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Switch perspective for opponent
    
    def get_action_probs(self, state, temperature=1):
        """Get normalized visit counts as action probabilities"""
        root = self.search(state)
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        moves = list(root.children.keys())
        
        if temperature == 0:  # Select most visited move
            action_idx = visit_counts.argmax()
            action_probs = np.zeros_like(visit_counts)
            action_probs[action_idx] = 1
        else:  # Sample move based on visit counts
            visit_counts = visit_counts ** (1 / temperature)
            action_probs = visit_counts / visit_counts.sum()
        
        return dict(zip(moves, action_probs))

class SelfPlay:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.mcts = MCTS(game, 
                        c_puct=args.get('c_puct', 1.0),
                        n_simulations=args.get('n_simulations', 800))
        self.temperature = args.get('temperature', 1.0)
        self.num_games = args.get('num_games', 100)
        
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
            if len(train_examples) == 10:  # After 10 moves
                temperature = 0.5
            elif len(train_examples) == 20:  # After 20 moves
                temperature = 0.25
        
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.get('learning_rate', 0.001))
    batch_size = args.get('batch_size', 32)
    
    # Convert examples to tensors
    states = torch.FloatTensor([ex['state'] for ex in examples])
    action_probs = torch.FloatTensor([list(ex['action_probs'].values()) for ex in examples])
    values = torch.FloatTensor([ex['value'] for ex in examples])
    
    dataset = torch.utils.data.TensorDataset(states, action_probs, values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    
    for batch_idx, (state, target_probs, target_value) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        start_probs, end_probs, value = model(state)
        
        # Calculate loss
        policy_loss = -(target_probs * torch.log(start_probs)).sum(dim=1).mean()
        policy_loss += -(target_probs * torch.log(end_probs)).sum(dim=1).mean()
        value_loss = ((value - target_value) ** 2).mean()
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
    
    model.eval()