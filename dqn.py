import torch
import torch.nn as nn
import torch.optim as optim
#from drl_env import ReplayBuffer
import random
import numpy as np

class RolloutBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, s0, a, r, s1, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0, a, r, s1, done))

    def sample(self, batch_size):
        #print(random.sample(self.buffer, batch_size))
        s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        s0 = torch.tensor(s0, dtype=torch.float).cuda()
        s1 = torch.tensor(s1, dtype=torch.float).cuda()
        a = torch.tensor(a, dtype=torch.long).cuda()
        r = torch.tensor(r, dtype=torch.float).cuda()
        done = torch.tensor(done, dtype=torch.float).cuda()
        return s0, a, r, s1, done

    def size(self):
        return len(self.buffer)

class DQNAgent(nn.Module):
    def __init__(self, state_size, action_size, device):
        super(DQNAgent, self).__init__()
        self.buffer = RolloutBuffer(2000)
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9
        self.learning_rate = 0.02
        size = 256
        #224:optimal:5/100,better than greedy:55/100, at least greedy:76/100,320.0211285299997s
        #280:optimal:12/100,better than greedy:64/100, at least greedy:86/100,434.8715323809997s

        self.nn = nn.Sequential(
            nn.Linear(self.state_size, size),
            nn.ReLU(),
            nn.Linear(size, 2*size),
            nn.ReLU(),
            #nn.Linear(2*size, 2*size),
            #nn.ReLU(),
            nn.Linear(2*size, size),
            nn.ReLU(),
            nn.Linear(size, self.action_size)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.85, 0.99), weight_decay=1e-4)#no need cuda
        self.is_training = True
        self.device = device

    def model(self, x):
        return self.nn(x)
    
    def act(self, obs, idx):
        state = torch.tensor(obs[idx], dtype=torch.float).cuda()
        #print(random.random())
        if random.random() > self.epsilon or not self.is_training:            
            q_value = self.model(state)
            #print(q_value.size())
            action = q_value.max(0)[1].item()
        else:
            action = random.randrange(self.action_size)
            #print(action)
        return action
    
    def remember(self, state, action, reward, next_state, done):#, batch_size):
        self.buffer.add(state, action, reward, next_state, done)
        #if self.memory.size() < batch_size:
            #self.memory.add(state, action, reward, next_state, done)

    def replay(self, batch_size):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        s0, a, r, s1, done = self.buffer.sample(batch_size)

        q_values = self.model(s0)
        next_q_values = self.model(s1)
        next_q_value = next_q_values.max(1)[0]
        
        #print(q_values.size(),a.unsqueeze(1).size())
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()