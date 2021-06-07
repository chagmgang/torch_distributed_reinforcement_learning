import copy
import torch
import numpy as np

from rl.loss.dqn import DQNLoss

def get_value(value, action):

    batch, action_size = value.shape
    onehot_action = torch.zeros_like(value)
    onehot_action[range(batch), action] = 1

    value = torch.sum(onehot_action * value, dim=1)
    return value

class DQNAgent:

    def __init__(self, model, action_size, device):

        self.main = model.to(device)
        self.target = copy.deepcopy(self.main)
        self.action_size = action_size
        self.device = device

        self.optim = torch.optim.Adam(
                self.main.parameters(), lr=1e-3)

        self.criterion = DQNLoss()

    def get_weights(self):
        return self.main.state_dict()

    def set_weights(self, param):
        self.main.load_state_dict()

    def get_action(self, x):
        x = torch.as_tensor([x]).to(torch.float).to(self.device)
        value = self.main(x)
        value = value.detach().cpu().numpy()[0]
        action = np.argmax(value)
        return action

    def td_error(self, state, next_state, action, reward, done):

        state = torch.as_tensor(state).to(torch.float).to(self.device)
        next_state = torch.as_tensor(next_state).to(torch.float).to(self.device)
        action = torch.as_tensor(action).to(torch.long).to(self.device)
        reward = torch.as_tensor(reward).to(torch.float).to(self.device)
        done = torch.as_tensor(done).to(torch.float).to(self.device)

        main_q_value = self.main(state)
        next_main_q_value = self.main(next_state)
        next_action = torch.argmax(next_main_q_value, dim=1)
        target_q_value = self.target(next_state)

        main_q_value = get_value(main_q_value, action)
        target_q_value = get_value(target_q_value, next_action)

        td_error = self.criterion.td_error(main_q_value, target_q_value, reward, done)
        td_error = np.abs(td_error.detach().cpu().numpy())
        return td_error

    def train(self, state, next_state, action, reward, done, is_weight):

        self.main.train()
        
        state = torch.as_tensor(state).to(torch.float).to(self.device)
        next_state = torch.as_tensor(next_state).to(torch.float).to(self.device)
        action = torch.as_tensor(action).to(torch.long).to(self.device)
        reward = torch.as_tensor(reward).to(torch.float).to(self.device)
        done = torch.as_tensor(done).to(torch.float).to(self.device)
        is_weight = torch.as_tensor(is_weight).to(torch.float).to(self.device)

        main_q_value = self.main(state)
        next_main_q_value = self.main(next_state)
        next_action = torch.argmax(next_main_q_value, dim=1)
        target_q_value = self.target(next_state)

        main_q_value = get_value(main_q_value, action)
        target_q_value = get_value(target_q_value, next_action)

        loss, td_error = self.criterion(main_q_value, target_q_value, reward, done, is_weight)

        td_error = np.abs(td_error.detach().cpu().numpy())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.main.eval()

        return loss, td_error

    def main_to_target(self):
        state_dict = self.main.state_dict()
        self.target.load_state_dict(state_dict)
