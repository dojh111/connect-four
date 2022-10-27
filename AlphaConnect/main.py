from minimalConnectFour import Board
from AlphaConnect import AlphaConnect
from trainer import Trainer
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = {
    'batch_size' : 64,
    'num_simulations':100,
    'numIters':10,
    'numEps':10,
    'epochs':25,
    'checkpoint':'latest.pt',
    'lr' : 0.001,
}

game = Board()
model = AlphaConnect().to(device)

trainer = Trainer(game, model, args, device)
trainer.learn()