import gym
import ray
import time
import torch
import multiprocessing

import numpy as np

from rl.buffer.apex import LocalBuffer
from rl.buffer.apex import Memory
from rl.agent.dqn import DQNAgent
from rl.model.cartpole import ValueModel

from tensorboardX import SummaryWriter
from threading import Thread

@ray.remote
class Actor(object):

    def __init__(self, actor_id, policy, buffer_size, batch_size):

        self.env = gym.make('CartPole-v0')
        self.local_buffer = LocalBuffer(int(buffer_size))
        self.id = actor_id
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.writer = SummaryWriter(f'runs/{self.id}')

        self.policy = DQNAgent(
                model=ValueModel(),
                action_size=2,
                device=torch.device('cpu'))

        self.running = Thread(
                target=self.runner)

        self.running.start()

    def set_weights(self, params):
        self.policy.set_weights(params)
        if len(self.local_buffer) == self.buffer_size:
            data = self.local_buffer.sample(self.batch_size)
        else:
            data = None

        info = {'id': self.id,
                'data': data}

        return None, info

    def runner(self):

        episode = 0

        while True:

            state = self.env.reset()
            done = False
            score = 0
            episode += 1
            epsilon = 1 / (episode * 0.05 + 1)

            while not done:

                action = self.policy.get_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)

                score += reward

                reward = 0
                if done:
                    if score == 200:
                        reward = 1
                    else:
                        reward = -1

                self.local_buffer.append(
                        state=state, next_state=next_state,
                        action=action, reward=reward,
                        done=done)

                state = next_state

            self.writer.add_scalar('data/score', score, episode)
            self.writer.add_scalar('data/epsilon', epsilon, episode)

            print(f'actor id : {self.id} | episode : {episode} | scoddre : {score} | eps : {epsilon}')
    

@ray.remote
class Learner(object):

    def __init__(self, policy, buffer_size, batch_size):

        self.replay_buffer = Memory(int(buffer_size))
        self.policy = policy

        self.writer = SummaryWriter('runs/learner')

        self.batch_size = batch_size
        self.train_step = 0
        self.replay_size = 0
        self.buffer_size = buffer_size

        self.running = Thread(
                target=self.train)

        self.running.start()

    def train(self):

        while True:

            if self.replay_size > self.batch_size:

                self.train_step += 1

                s = time.time()

                minibatch, idxs, is_weight = self.replay_buffer.sample(self.batch_size)
                minibatch = np.stack(minibatch)
                
                state = np.stack(minibatch[:, 0])
                next_state = np.stack(minibatch[:, 1])
                action = np.stack(minibatch[:, 2])
                reward = np.stack(minibatch[:, 3])
                done = np.stack(minibatch[:, 4])

                loss, td_error = self.policy.train(
                        state, next_state, action, reward, done, is_weight)

                self.writer.add_scalar('data/loss', float(loss), self.train_step)
                self.writer.add_scalar('data/time', time.time() - s, self.train_step)

                for i in range(len(idxs)):
                    self.replay_buffer.update(idxs[i], td_error[i])

                if self.train_step % 100 == 0:
                    self.policy.main_to_target()

    def get_weights(self):
        return self.policy.get_weights()

    def append(self, state, next_state, action,
               reward, done):

        td_error = self.policy.td_error(
                np.stack(state), np.stack(next_state),
                np.stack(action), np.stack(reward), np.stack(done))

        for i in range(len(state)):

            if self.replay_size > self.buffer_size:
                self.replay_size = self.buffer_size
            else:
                self.replay_size += 1
            self.replay_buffer.add(
                    td_error[i],
                    [state[i], next_state[i],
                     action[i], reward[i], done[i]])


def main(num_workers):

    batch_size = 64
    buffer_size = 1e5

    policy = DQNAgent(
            model=ValueModel(),
            action_size=2,
            device=torch.device('cpu'))

    learner = Learner.remote(
            policy=policy,
            buffer_size=buffer_size,
            batch_size=batch_size)

    agents = [
            Actor.remote(
                actor_id=i,
                policy=DQNAgent(
                    model=ValueModel(),
                    action_size=2,
                    device=torch.device('cpu')),
                buffer_size=2*batch_size,
                batch_size=batch_size,
                ) for i in range(num_workers)]

    params = [learner.get_weights.remote()]
    done_id, params = ray.wait(params)
    params = ray.get(done_id)[0]
    samples = [agent.set_weights.remote(params) for agent in agents]

    while True:

        done_id, samples = ray.wait(samples)
        sample, info = ray.get(done_id)[0]
        if info['data'] is not None:
            data = info['data']
            learner.append.remote(
                    state=data.state,
                    next_state=data.next_state,
                    action=data.action,
                    reward=data.reward,
                    done=data.done)

        params = [learner.get_weights.remote()]
        done_id, params = ray.wait(params)
        params = ray.get(done_id)[0]
        samples.extend([
            agents[info['id']].set_weights.remote(params)])


if __name__ == '__main__':
    ray.init()
    cpu_count = multiprocessing.cpu_count()
    main(num_workers=4)
