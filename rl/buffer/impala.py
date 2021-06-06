import collections
import numpy as np

class Trajectory:

    def __init__(self, state, next_state, action,
                 reward, done, mu):

        self.state = state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done
        self.mu = mu

class LocalBuffer:

    def __init__(self, trajectory):

        self.state = collections.deque(maxlen=trajectory)
        self.next_state = collections.deque(maxlen=trajectory)
        self.action = collections.deque(maxlen=trajectory)
        self.reward = collections.deque(maxlen=trajectory)
        self.done = collections.deque(maxlen=trajectory)
        self.mu = collections.deque(maxlen=trajectory)

    def append(self, state, next_state, action, reward, done, mu):

        self.state.append(state)
        self.next_state.append(next_state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)
        self.mu.append(mu)

    def __len__(self):
        return len(self.state)

    def sample(self):
        return Trajectory(
                np.stack(self.state),
                np.stack(self.next_state),
                np.stack(self.action),
                np.stack(self.reward),
                np.stack(self.done),
                np.stack(self.mu))

class ImpalaBuffer:

    def __init__(self, trajectory, buffer_size):

        self.trajectory = trajectory
        self.traj = LocalBuffer(trajectory)
        
        self.state = collections.deque(maxlen=buffer_size)
        self.next_state = collections.deque(maxlen=buffer_size)
        self.action = collections.deque(maxlen=buffer_size)
        self.reward = collections.deque(maxlen=buffer_size)
        self.done = collections.deque(maxlen=buffer_size)
        self.mu = collections.deque(maxlen=buffer_size)

    def append(self, state, next_state, action,
               reward, done, mu):

        self.traj.append(state, next_state,
                         action, reward, done, mu)

        if len(self.traj) == self.trajectory:
            traj = self.traj.sample()
            self.state.append(np.stack(traj.state))
            self.next_state.append(np.stack(traj.next_state))
            self.action.append(np.stack(traj.action))
            self.reward.append(np.stack(traj.reward))
            self.done.append(np.stack(traj.done))
            self.mu.append(np.stack(traj.mu))

    def __len__(self):
        return len(self.state)

    def sample(self, batch):
        
        latest_state = self.get_latest_data(self.state, int(batch / 2))
        latest_next_state = self.get_latest_data(self.next_state, int(batch / 2))
        latest_action = self.get_latest_data(self.action, int(batch / 2))
        latest_reward = self.get_latest_data(self.reward, int(batch / 2))
        latest_done = self.get_latest_data(self.done, int(batch / 2))
        latest_mu = self.get_latest_data(self.mu, int(batch / 2))

        arange = np.arange(len(self.state) - int(batch / 2))
        np.random.shuffle(arange)
        batch_idx = arange[:int(batch / 2)]
        
        old_state = self.get_old_data(self.state, batch_idx)
        old_next_state = self.get_old_data(self.next_state, batch_idx)
        old_action = self.get_old_data(self.action, batch_idx)
        old_reward = self.get_old_data(self.reward, batch_idx)
        old_done = self.get_old_data(self.done, batch_idx)
        old_mu = self.get_old_data(self.mu, batch_idx)

        return Trajectory(
                state=self.extend(old_state, latest_state),
                next_state=self.extend(old_next_state, latest_next_state),
                action=self.extend(old_action, latest_action),
                reward=self.extend(old_reward, latest_reward),
                done=self.extend(old_done, latest_done),
                mu=self.extend(old_mu, latest_mu))

    def extend(self, old, latest):
        l = list()
        l.extend(old)
        l.extend(latest)
        return np.stack(l)


    def get_old_data(self, deque, batch_idx):
        old = [deque[i] for i in batch_idx]
        return old

    def get_latest_data(self, deque, size):
        latest = [deque[-(i+1)] for i in range(size)]
        return np.stack(latest)

class GlobalBuffer(ImpalaBuffer):

    def append(self, state, next_state, action, reward, done, mu):

        self.state.append(np.stack(state))
        self.next_state.append(np.stack(next_state))
        self.action.append(np.stack(action))
        self.reward.append(np.stack(reward))
        self.done.append(np.stack(done))
        self.mu.append(mu)
