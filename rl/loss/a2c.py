import torch
import torch.nn as nn

class A2CLoss(nn.Module):

    def __init__(self, discount=0.99):
        super(A2CLoss, self).__init__()

        self.discount = discount

    def get_pi(self, pi, action):

        batch, action_size = pi.shape
        onehot_action = torch.zeros_like(pi)
        onehot_action[range(batch), action] = 1

        pi = pi * onehot_action
        
        return torch.sum(pi, dim=1)

    def forward(self, pi, value, next_value,
                reward, done, action):

        pi = self.get_pi(pi, action)
        next_value = next_value.detach()[:, 0]
        value = value[:, 0]

        log_pi = torch.log(pi)
        target_value = reward + self.discount * (1 - done) * next_value

        value_loss = torch.mean((target_value - value) ** 2)
        pi_loss = torch.mean(log_pi * (target_value - value).detach())

        tot_loss = value_loss - pi_loss

        return tot_loss
