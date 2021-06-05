from rl.loss.a2c import A2CLoss

import torch
import torch.nn as nn
import numpy as np

class A2CAgent:

    def __init__(self, model, action_size, device):
        
        self.model = model.to(device)
        self.action_size = action_size
        self.device = device

        self.optim = torch.optim.Adam(
                self.model.parameters(), lr=0.001)

        self.criterion = A2CLoss()

        self.model.eval()

    def get_action(self, x):
        x = torch.as_tensor([x]).to(torch.float).to(self.device)
        prob, _ = self.model(x)
        prob = prob.detach().cpu().numpy()[0]
        action = np.random.choice(self.action_size, p=prob)
        return action

    def train(self, state, next_state, reward, done, action):

        state = torch.as_tensor(state).to(torch.float)
        next_state = torch.as_tensor(next_state).to(torch.float)
        reward = torch.as_tensor(reward).to(torch.float)
        done = torch.as_tensor(done).to(torch.float)
        action = torch.as_tensor(action).to(torch.long)

        self.model.train()

        pi, value = self.model(state)
        _, next_value = self.model(next_state)

        loss = self.criterion(
                pi, value, next_value, reward, done, action)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.model.eval()
