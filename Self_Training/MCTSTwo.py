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
        
        print_once = True
        
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
                valid_moves = self.game.get_valid_moves(game_state, print_once)
                print_once = False
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

