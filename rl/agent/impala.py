import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from rl.loss.vtrace import ImpalaLoss


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())

class ImpalaAgent:

    def __init__(self, model, action_size, device):
        
        self.model = model.to(device)
        self.action_size = action_size
        self.device = device

        self.optim = torch.optim.RMSprop(
                self.model.parameters(), lr=0.00048)

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

        '''
        discounts = (1 - done).float() * self.discount

        actor_outputs = torch.transpose(mu, 0, 1)
        learner_outputs = torch.transpose(pi, 0, 1)
        actions = torch.transpose(action, 0, 1)
        discounts = torch.transpose(discounts, 0, 1)
        rewards = torch.transpose(reward, 0, 1)
        values = torch.transpose(values, 0, 1)
        bootstrap_value = torch.transpose(next_values, 0, 1)[-1]
        
        vtrace_returns = from_logits(
                behavior_policy_logits=actor_outputs,
                target_policy_logits=learner_outputs,
                actions=actions,
                discounts=discounts,
                rewards=rewards,
                values=values,
                bootstrap_value=bootstrap_value)

        pg_loss = compute_policy_gradient_loss(
                learner_outputs,
                actions,
                vtrace_returns.pg_advantages)

        baseline_loss = compute_baseline_loss(
                vtrace_returns.vs - values)

        entropy_loss = compute_entropy_loss(
            learner_outputs
        )

        total_loss = pg_loss + baseline_loss * 0.5 + entropy_loss * 0.0006
        '''

        total_loss, pg_loss, baseline_loss, entropy_loss = self.criterion(
                behavior_policy_logits=mu,
                target_policy_logits=pi,
                action=action, values=values,
                next_values=next_values, reward=reward,
                done=done)

        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()

        self.model.eval()

        return pg_loss, baseline_loss, entropy_loss
