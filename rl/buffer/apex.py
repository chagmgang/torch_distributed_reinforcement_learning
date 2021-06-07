import random
import collections
import numpy as np

class Sample:

    def __init__(self, state, next_state, action,
                 reward, done):

        self.state = state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done

class LocalBuffer:

    def __init__(self, capacity):

        self.state = collections.deque(maxlen=capacity)
        self.next_state = collections.deque(maxlen=capacity)
        self.action = collections.deque(maxlen=capacity)
        self.reward = collections.deque(maxlen=capacity)
        self.done = collections.deque(maxlen=capacity)

    def append(self, state, next_state, action,
               reward, done):

        self.state.append(state)
        self.next_state.append(next_state)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)

    def __len__(self):
        return len(self.state)

    def sample(self, batch_size):
        arange = np.arange(len(self.state))
        np.random.shuffle(arange)
        batch_idx = arange[:batch_size]
        
        state = [self.state[b] for b in batch_idx]
        next_state = [self.next_state[b] for b in batch_idx]
        action = [self.action[b] for b in batch_idx]
        reward = [self.reward[b] for b in batch_idx]
        done = [self.done[b] for b in batch_idx]

        return Sample(
                np.stack(state), np.stack(next_state),
                np.stack(action), np.stack(reward),
                np.stack(done))

class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class Memory(object):
    e = 0.001
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def reset(self):
        self.tree = SumTree(self.capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a, b = segment * i, segment * (i+1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)
