import torch
import torch.nn as nn

class DQNLoss(nn.Module):

    def __init__(self, discount_factor=0.99):
        super(DQNLoss, self).__init__()

        self.discount = discount_factor

    def td_error(self, main_value, target_value,
                 reward, done):
        target = target_value * (1 - done) * self.discount + reward
        target = target.detach() - main_value

        return target

    def forward(self, main_value, target_value,
                reward, done, is_weight):

        target = self.td_error(
                main_value, target_value, reward, done)

        error = target ** 2
        loss = torch.mean(error * is_weight)

        return loss, target
