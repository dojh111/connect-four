import numpy as np
from copy import deepcopy
import torch


class Board:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.player = 1 # Can be 1 or -1
        self.column_heights = np.ones(7, dtype=int) * 5
        self.actions = list(range(7))
        self.last_action = None
        
    def play_action(self, action):
        if action not in self.actions:
            raise ValueError("Invalid action")
        else:
            new_board = deepcopy(self)
            
            # Update board
            new_board.board[new_board.column_heights[action], action] = new_board.player
            
            # Update column heights
            new_board.column_heights[action] -= 1
            
            # Update actions
            if new_board.column_heights[action] < 0:
                new_board.actions.remove(action)
            
            # Update player
            new_board.player *= -1
            
            new_board.last_action = action
            
            return new_board
    
    def get_actions(self):
        return self.actions
    
    def check_horizontal(self, player):
        for i in range(6):
            for j in range(4):
                if np.all(self.board[i, j:j+4] == player):
                    return True
        return False
    
    def check_vertical(self, player):
        for i in range(3):
            for j in range(7):
                if np.all(self.board[i:i+4, j] == player):
                    return True
        return False
    
    def check_diagonals(self, player):
        for i in range(3):
            for j in range(4):
                if np.all(self.board[i:i+4, j:j+4].diagonal() == player):
                    return True
        for i in range(3):
            for j in range(3, 7):
                if np.all(np.fliplr(self.board[i:i+4, j-3:j+1]).diagonal() == player):
                    return True
        return False
    
    def check_winner(self):
        return self.check_horizontal(self.player*-1) or self.check_vertical(self.player*-1) or self.check_diagonals(self.player*-1)
    
    def get_score(self):
        if self.isDone():
            if self.check_winner():
                return self.player*-1
        else:
            return 0
    
    def isDone(self):
        if self.last_action == None:
            return False
        return len(self.actions) == 0 or self.check_winner() != 0
    
    def get_canonical_form(self):
        encoded = np.zeros([6,7,3]).astype(int)
         
        for row in range(6):
            for col in range(7):
                if self.board[row,col] == 1:
                    encoded[row,col,0] = 1
                elif self.board[row,col] == -1:
                    encoded[row,col,1] = 1
        
        encoded[:,:,2] = self.player
        
        encoded = np.transpose(encoded, (2,0,1))
        encoded = np.expand_dims(encoded, axis=0)
        
        return torch.Tensor(encoded)
    
    def print_board(self):
        temp = np.where(self.board == 0, ' ', np.where(self.board == 1, 'X', 'O'))
        temp_board = list(temp)
        temp_board.insert(0, list(range(7)))
        
        def format_row(row):
            return '|' + '|'.join('{0}'.format(x) for x in row) + '|'
        
        def format_board(board):
            return '\n\n'.join(format_row(row) for row in board)
        
        print("\n\n" + format_board(temp_board) + "\n\n")
        
        
if __name__ == "__main__":
    board = Board()
    board.print_board()
    
    print(board.get_canonical_form())
        
        
        