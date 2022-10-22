import collections
import numpy as np
import random
import math
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = './model.pt'

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.001
gamma         = 0.98
buffer_limit  = 5000
batch_size    = 32
max_episodes  = 2000
t_max         = 600
min_buffer    = 1000
target_update = 20 # episode(s)
train_steps   = 10
max_epsilon   = 1.0
min_epsilon   = 0.01
epsilon_decay = 500
print_interval= 20


NetworkInput = collections.namedtuple('NetworkInput', ('one_array', 'minus_one_array', 'turn_array'))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.buffer = np.empty(buffer_limit, dtype=NetworkInput)
        self.buffer_limit = buffer_limit
        self.counter =  0
        self.num_samples = 0

    def push(self, transition):
        '''
        Input:
            * `NetworkInput` (`NetworkInput`): tuple representing a single state (see `NetworkInput` above).

        Output:
            * None
        '''
        if (self.counter + 1 > self.buffer_limit):
            self.counter = 0
        
        if (self.num_samples < self.buffer_limit):
            self.num_samples += 1
            
        self.buffer[self.counter] = transition
        self.counter += 1

            
    
    def sample(self, batch_size):
        '''
        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device `device`.
        '''
        sample_ind = np.random.randint(0, high=self.__len__(), size=batch_size)
        
        batch = self.buffer[sample_ind]
        
        states_size = batch[0].state.shape
        
        states = torch.zeros(batch_size, states_size[0], states_size[1], states_size[2], dtype=torch.float, device=device)
        actions = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        rewards = torch.zeros(batch_size, 1, dtype=torch.float, device=device)
        next_states = torch.zeros(batch_size, states_size[0], states_size[1], states_size[2], dtype=torch.float, device=device)
        dones = torch.zeros(batch_size, 1, dtype=torch.float, device=device)
        
        for i in range(batch_size):
            states[i] = torch.tensor(batch[i].state, dtype=torch.float, device=device)
            actions[i] = torch.tensor(batch[i].action, dtype=torch.long, device=device)
            rewards[i] = torch.tensor(batch[i].reward, dtype=torch.float, device=device)
            next_states[i] = torch.tensor(batch[i].next_state, dtype=torch.float, device=device)
            dones[i] = torch.tensor(batch[i].done, dtype=torch.float, device=device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return self.num_samples
    
    
class Base(nn.Module):
    '''
    Base neural network model that handles dynamic architecture.
    '''
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.construct()

    def construct(self):
        raise NotImplementedError

    def forward(self, x):
        if hasattr(self, 'features'):
            x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
    
    def feature_size(self):
        x = autograd.Variable(torch.zeros(1, *self.input_shape))
        if hasattr(self, 'features'):
            x = self.features(x)
        return x.view(1, -1).size(1)
    


class ConvolutionalBlock(nn.Module):
    def __init__(self):
        super(ConvolutionalBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s
    
class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3*6*7, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6*7*32, 7)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 3*6*7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 6*7*32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        
        return p, v

class ConnectNet(nn.Module):
    def __init__(self):
        super(ConnectNet, self).__init__()
        self.conv = ConvolutionalBlock()
        for block in range(19):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
    
class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2 # MSE loss
        policy_error = torch.sum((-policy* 
                                (1e-8 + y_policy.float()).float().log()), 1) # cross entropy loss
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error
    
if __name__ == "__main__":
    net = ConnectNet()
    test = torch.rand(1,3,6,7)
    
    p,v = net(test)
    
    print(p)
    print(v)