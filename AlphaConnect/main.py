from MCTS import MonteCarloTreeSearch
from AlphaConnect import AlphaConnect, ReplayBuffer, AlphaLoss
from minimalConnectFour import Board
import torch
import torch.optim as optim
from tqdm import tqdm


if __name__ == "__main__":
    board = Board()
    network = AlphaConnect()
    replay_buffer = ReplayBuffer(1000)
    
    
    for i in tqdm(range(10)):
        mcts = MonteCarloTreeSearch(100, 1, network)
        action = mcts.buildTreeAndReturnBestAction(board)
        nodes = mcts.getStatePriorValues(mcts.root)
        
        for node in nodes:
            replay_buffer.push((node[0], node[1], node[2]))
            
        
        cuda = torch.cuda.is_available()
        network.train()
        
        criterion = AlphaLoss()
        optimizer = optim.Adam(network.parameters(), lr=0.001, betas=(0.8, 0.999))
        
        loss_per_epoch = []
        for epoch in range(10):
            total_loss = 0
            batch = replay_buffer.sample(32)
            for i in range(len(batch)):
                states = torch.tensor(batch[i][0]).float()
                priors = torch.tensor(batch[i][1]).float()
                values = torch.tensor(batch[i][2]).float()
                
                
                #if cuda:
                #    states = states.cuda()
                #    priors = priors.cuda()
                
                #    values = values.cuda()
                
                pred_prior, pred_value = network(states)
                
                loss = criterion(pred_prior, pred_value, priors, values)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            loss_per_epoch.append(total_loss/len(batch))
            
            print("Epoch: {}, Loss: {}".format(epoch, total_loss/len(batch)))    
