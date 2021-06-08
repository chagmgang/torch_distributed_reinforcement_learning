import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rl.loss.vtrace import ImpalaLoss


class ImpalaAgent:

    def __init__(self, model, action_size, device):
        
        self.model = model.to(device)
        self.action_size = action_size
        self.device = device

        self.optim = torch.optim.RMSprop(
                self.model.parameters(),
                lr=0.00048,
                momentum=0,
                eps=0.01,
                alpha=0.99)

        self.criterion = ImpalaLoss()
        self.discount = 0.99
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

        state = torch.as_tensor(state).to(torch.float).to(self.device)
        next_state = torch.as_tensor(next_state).to(torch.float).to(self.device)
        reward = torch.as_tensor(reward).to(torch.float).to(self.device)
        done = torch.as_tensor(done).to(torch.float).to(self.device)
        action = torch.as_tensor(action).to(torch.long).to(self.device)
        mu = torch.as_tensor(mu).to(torch.float).to(self.device)

        self.model.train()

        pi, value = [], []
        for s in state:
            p, v = self.model.get_logit(s)
            pi.append(p)
            value.append(v)
        pi = torch.stack(pi)
        values = torch.stack(value)[:, :, 0]

        next_values = []
        for ns in next_state:
            _, v = self.model.get_logit(ns)
            next_values.append(v)
        next_values = torch.stack(next_values)[:, :, 0]

        total_loss, pg_loss, baseline_loss, entropy_loss = self.criterion(
                behavior_policy_logits=mu,
                target_policy_logits=pi,
                action=action, values=values,
                next_values=next_values, reward=reward,
                done=done)

        self.optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
        self.optim.step()

        self.model.eval()

        return pg_loss, baseline_loss, entropy_loss
