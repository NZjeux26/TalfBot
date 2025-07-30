import math
import numpy as np

class MCTSNode:
    def __init__(self, game, state, parent=None, prior_prob=0):
        self.game = game
        self.state = state
        self.parent = parent
        self.prior_prob = prior_prob
        
        self.children = {}  # map of moves to nodes
        self.valid_moves = game.get_valid_moves(state)
        self.move_probs = game.get_move_probabilities(state, self.valid_moves)
        
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False
        
class MCTS:
    def __init__(self, game, num_simulations=800, c_puct=1.0):
        self.game = game
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(self, root_state):
        root = MCTSNode(self.game, root_state)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.is_expanded:
                action, node = self.select_child(node)
                search_path.append(node)
            
            # Expansion and Evaluation
            value = self.expand_and_evaluate(node)
            
            # Backup
            self.backup(search_path, value)
        
        # Return move probabilities based on visit counts
        return self.get_action_probs(root)
    
    def select_child(self, node):
        """Select the child with the highest UCB score"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for move, child in node.children.items():
            ucb_score = self.get_ucb_score(node, child)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = move
                best_child = child
        
        return best_action, best_child
    
    def get_ucb_score(self, parent, child):
        """Calculate UCB score for a child node"""
        prior_score = (self.c_puct * child.prior_prob * 
                      math.sqrt(parent.visit_count) / (1 + child.visit_count))
        value_score = -child.value_sum / child.visit_count if child.visit_count > 0 else 0
        return value_score + prior_score
    
    def expand_and_evaluate(self, node):
        """Expand node and return value from neural network"""
        # Get value from neural network
        _, _, value = self.game.get_policy_value_predictions(node.state)
        
        # If game is ended, use true game value
        game_value = self.game.get_game_ended(node.state)
        if game_value != 0:
            return game_value
        
        # Create children for all valid moves
        for move in node.valid_moves:
            next_state = self.game.make_move(node.state, move)
            child = MCTSNode(
                self.game,
                next_state,
                parent=node,
                prior_prob=node.move_probs.get(move, 1e-8)
            )
            node.children[move] = child
        
        node.is_expanded = True
        return value
    
    def backup(self, search_path, value):
        """Update statistics of all nodes in search path"""
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Value flips for opposing player
    
    def get_action_probs(self, root, temperature=1.0):
        """Returns probabilities for each move based on visit counts"""
        visits = {move: child.visit_count for move, child in root.children.items()}
        
        if temperature == 0:  # Pick move with highest visit count
            max_visit = max(visits.values())
            moves = [move for move, visit in visits.items() if visit == max_visit]
            probs = {move: 1.0/len(moves) if move in moves else 0.0 
                    for move in root.valid_moves}
        else:
            # Apply temperature to visit count distribution
            counts = np.array([visits.get(move, 0) for move in root.valid_moves])
            if temperature == 1.0:
                probs = counts / counts.sum()
            else:
                counts = counts ** (1.0 / temperature)
                probs = counts / counts.sum()
            probs = {move: prob for move, prob in zip(root.valid_moves, probs)}
            
        return probs