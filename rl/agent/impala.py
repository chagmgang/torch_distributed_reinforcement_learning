from rl.loss.impala import from_softmax
from rl.loss.impala import split_data
from rl.loss.impala import compute_policy_gradient_loss
from rl.loss.impala import compute_baseline_loss

import torch
import torch.nn as nn
import numpy as np

class ImpalaAgent:

    def __init__(self, model, action_size, device):
        
        self.model = model.to(device)
        self.action_size = action_size
        self.device = device

        self.optim = torch.optim.Adam(
                self.model.parameters(), lr=0.001)

        self.model.eval()

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

        discounts = (1 - done) * 0.99

        first_pi, second_pi, third_pi = split_data(pi)
        first_mu, second_mu, third_mu = split_data(mu)
        first_action, second_action, third_action = split_data(action)
        first_v, second_v, third_v = split_data(values)
        first_nv, second_nv, third_nv = split_data(next_values)
        first_r, second_r, third_r = split_data(reward)
        first_d, second_d, third_d = split_data(discounts)

        vs, clipped_pg_rhos = from_softmax(
                behavior_policy=first_mu, target_policy=first_pi,
                actions=first_action, discounts=first_d,
                rewards=first_r, values=first_v,
                next_values=first_nv)

        vs_plus_1, _ = from_softmax(
                behavior_policy=second_mu, target_policy=second_pi,
                actions=second_action, discounts=second_d,
                rewards=second_r, values=second_v,
                next_values=second_nv)

        pg_advantage = clipped_pg_rhos * \
                (first_r + first_d * vs_plus_1 - first_v)

        pi_loss = compute_policy_gradient_loss(
                first_pi, first_action, pg_advantage)

        value_loss = compute_baseline_loss(
                vs, first_v)

        tot_loss = pi_loss + value_loss
        self.optim.zero_grad()
        tot_loss.backward()
        self.optim.step()

        self.model.eval()

        '''

        self.model.train()

        pi, value = self.model(state)
        _, next_value = self.model(next_state)

        loss = self.criterion(
                pi, value, next_value, reward, done, action)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.model.eval()
        '''
