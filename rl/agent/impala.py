import torch
import torch.nn as nn
import numpy as np

from rl.loss.impala import ImpalaLoss

class ImpalaAgent:

    def __init__(self, model, action_size, device):
        
        self.model = model.to(device)
        self.action_size = action_size
        self.device = device

        self.optim = torch.optim.RMSprop(
                self.model.parameters(), lr=1e-3)

        self.criterion = ImpalaLoss()

        self.model.eval()

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, param):
        self.model.load_state_dict(param)

    def get_action(self, x):
        x = torch.as_tensor([x]).to(torch.float).to(self.device)
        prob, _ = self.model(x)
        prob = prob.detach().cpu().numpy()[0]
        action = np.random.choice(self.action_size, p=prob)
        return action, prob

    def train(self, state, next_state, reward, done, action, mu):

        state = torch.as_tensor(state).to(torch.float)
        next_state = torch.as_tensor(next_state).to(torch.float)
        reward = torch.as_tensor(reward).to(torch.float)
        done = torch.as_tensor(done).to(torch.float)
        action = torch.as_tensor(action).to(torch.long)
        mu = torch.as_tensor(mu).to(torch.float)

        self.model.train()

        pi, value = [], []
        for s in state:
            p, v = self.model(s)
            pi.append(p)
            value.append(v)
        pi = torch.stack(pi)
        values = torch.stack(value)[:, :, 0]

        next_values = []
        for ns in next_state:
            _, v = self.model(ns)
            next_values.append(v)
        next_values = torch.stack(next_values)[:, :, 0]

        tot_loss, pi_loss, value_loss, ent = self.criterion(
                pi, mu, action,
                values, next_values,
                reward, done)
        self.optim.zero_grad()
        tot_loss.backward()
        self.optim.step()

        self.model.eval()

        return pi_loss, value_loss, ent
