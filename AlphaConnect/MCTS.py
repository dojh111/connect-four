import numpy as np
import math
from minimalConnectFour import Board
from AlphaConnect import AlphaConnect
from tqdm import tqdm
import collections


class Node():
    def __init__(self, state, move, parent=None):
        self.state = state
        self.move = move
        self.is_expanded  = False
        self.parent = parent
        self.children = dict()
        self.child_priors = np.zeros([7], dtype=np.float32)
        self.child_total_value = np.zeros([7], dtype=np.float32)
        self.child_number_visits = np.zeros([7], dtype=np.float32)
        self.action_idxes = []
    
    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value
        
    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]
    
    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value
        
    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)
    
    def child_U(self):
        return abs(self.child_priors) * math.sqrt(self.number_visits) / (1 + self.child_number_visits)
    
    def best_child(self):
        if self.action_idxes != []:
            best_moves = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(best_moves[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove
    
    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current
    
    def add_exploration_noise(self, action_idx, child_priors):
        valid_child_priors = child_priors[action_idx]
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(np.zeros([len(valid_child_priors)], dtype=np.float32) + 192)
        
        child_priors[action_idx] = valid_child_priors
        return child_priors
    
    def expand(self, child_priors):
        self.is_expanded = True
        action_idx = self.state.get_actions()
        c_p = child_priors
        
        if action_idx == []:
            self.is_expanded = False
        
        self.action_idxes = action_idx
        c_p[[i for i in range(len(child_priors)) if i not in action_idx]] = 0
        
        if self.parent.parent == None:
            c_p = self.add_exploration_noise(action_idx, c_p)
        self.child_priors = c_p
        
    def maybe_add_child(self, move):
        if move not in self.children:
            next_state = self.state.play_action(move)
            self.children[move] = Node(next_state, move, self)
        return self.children[move]
    
    def backup(self, value):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.state.player == 1:
                current.total_value += value
            elif current.state.player == -1:
                current.total_value -= value
        current.number_visits += 1
        current = current.parent
        
class DummyNode():
    def __init__(self):
        self.parent = None
        self.child_total_values = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        
def UCT_search(root, num_readouts, net, temp):
    root = Node(root, move=None, parent=DummyNode())
    
    for i in range(num_readouts):
        leaf = root.select_leaf()
        canonical_board = leaf.state.canonical_board().cuda()
        
        child_priors, value_estimate = net(canonical_board)
        
        child_priors = child_priors.detach().cpu().numpy().reshape(-1)
        value_estimate = value_estimate.detach().cpu().item()
        
        if leaf.state.is_terminal():
            leaf.backup(value_estimate)
            continue
        
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
        
    return root

def get_policy(root, temp):
    return ((root.child_number_visits)**(1/temp))/sum(root.child_number_visits**(1/temp))

def MCTS_self_play(net, num_readouts, temp, num_games):
    game_history = []
    for i in tqdm(range(num_games)):
        game = Board()
        root = Node(game, move=None, parent=DummyNode())
        while not game.is_terminal():
            root = UCT_search(root, num_readouts, net, temp)
            policy = get_policy(root, temp)
            move = np.random.choice(7, p=policy)
            game = game.play_action(move)
        game_history.append(game)
    return game_history
    
        
        