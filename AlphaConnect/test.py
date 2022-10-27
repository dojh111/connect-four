from minimalConnectFour import Board
from AlphaConnect import AlphaConnect
import torch
from MCTS import UCT_search, get_policy
import numpy as np

if __name__ == "__main__":
    board = Board()
    
    network = torch.load('./network.pt')
    network = network.cuda()
    
    turn = 0    
    while not board.isDone():
        board.print_board()
        if turn == 0:
            root = UCT_search(board, 100, network, 1)
            policy = get_policy(root, 1)
            board = board.play_action(np.random.choice(7, p=policy))
        else:
            move = int(input('Enter move: '))
            board = board.play_action(move)
        turn = 1 - turn