import ray
import gym
import time
import torch
import multiprocessing

from rl.model.breakout import Model
from rl.buffer.impala import LocalBuffer
from rl.buffer.impala import GlobalBuffer
from rl.agent.impala import ImpalaAgent
from env.map_wrappers import make_float_env

from tensorboardX import SummaryWriter
from collections import OrderedDict
from threading import Thread

import numpy as np

def convert_gpu_to_cpu(params):
    new_params = OrderedDict()
    for key, value in params.items():
        new_params[key] = value.to('cpu')
    return new_params

class Trajectory:

    def __init__(self, state, next_state, reward,
                 done, mu, action):

        self.state = state
        self.next_state = next_state
        self.reward = reward
        self.done = done
        self.mu = mu
        self.action = action


@ray.remote(num_cpus=1, memory=1024*1024*256)
class Actor(object):

    def __init__(self, actor_id, policy, trajectory):

        self.id = actor_id
        self.trajectory = trajectory

        self.policy = policy

        self.writer = SummaryWriter(f'runs/{self.id}')

        self.episode = 0
        self.start_env()

    def start_env(self):

        self.env = make_float_env("BreakoutDeterministic-v4")
        self.state = self.env.reset()
        self.done = False
        self.episode += 1
        self.episode_step = 0
        self.score = 0
        self.lives = 5
        self.prob = 0

    def rollout(self, params):

        self.policy.set_weights(params)

        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        mus = []

        for t in range(self.trajectory):
            
            self.action, self.mu = self.policy.get_action(self.state)
            self.next_state, self.reward, self.done, self.info = self.env.step(self.action)

            self.score += self.reward
            self.episode_step += 1
            self.prob += self.mu[self.action]

            if self.lives != self.info['ale.lives']:
                self.r = -1
                self.d = True
            else:
                self.r = self.reward
                self.d = False

            states.append(self.state)
            next_states.append(self.next_state)
            actions.append(self.action)
            rewards.append(self.r)
            dones.append(self.d)
            mus.append(self.mu)

            self.state = self.next_state

            if self.done:
                self.writer.add_scalar('data/score', self.score, self.episode)
                self.writer.add_scalar('data/episode_step', self.episode_step, self.episode)
                self.writer.add_scalar('data/prob', self.prob / self.episode_step, self.episode)
                print(f'actor : {self.id} | episode : {self.episode} | score : {self.score}')
                self.state = self.env.reset()
                self.done = False
                self.score = 0
                self.episode += 1
                self.episode_step = 0
                self.lives = 5
                self.prob = 0

        return Trajectory(
                state=np.stack(states), next_state=np.stack(next_states),
                action=np.stack(actions), reward=np.stack(rewards),
                done=np.stack(dones), mu=np.stack(mus)), {'id': self.id}

@ray.remote(num_gpus=1, memory=1024*1024*1024*4)
class Learner(object):

    def __init__(self, policy, trajectory, buffer_size, batch_size):

        self.writer = SummaryWriter('runs/learner')
        self.policy = policy
        self.global_buffer = GlobalBuffer(
                trajectory=trajectory,
                buffer_size=buffer_size)

        self.batch_size = batch_size
        self.train_step = 0

        self.thread = Thread(
                target=self.train)
        self.thread.start()

    def train(self):

        while True:

            if len(self.global_buffer) > 2 * self.batch_size:

                s = time.time()
                self.train_step += 1
                sample = self.global_buffer.sample(self.batch_size)
                p_loss, v_loss, ent = self.policy.train(
                        state=sample.state, next_state=sample.next_state,
                        reward=sample.reward, done=sample.done,
                        action=sample.action, mu=sample.mu)

                self.writer.add_scalar('data/pi_loss', float(p_loss), self.train_step)
                self.writer.add_scalar('data/value_loss', float(v_loss), self.train_step)
                self.writer.add_scalar('data/ent', float(ent), self.train_step)
                self.writer.add_scalar('data/time', time.time() - s, self.train_step)
            

    def get_weights(self):
        return self.policy.get_weights()

    def append(self, state, next_state, reward,
               done, action, mu):

        self.global_buffer.append(
                state=state, next_state=next_state,
                reward=reward, done=done,
                action=action, mu=mu)


def main(num_workers):

    trajectory = 20
    batch_size = 32
    buffer_size = 10 * batch_size
    action_size = 4

    writer = SummaryWriter('runs/learner')

    learner = Learner.remote(
            policy=ImpalaAgent(
                model=Model(),
                action_size=action_size,
                device=torch.device(0)),
            trajectory=trajectory,
            buffer_size=buffer_size,
            batch_size=batch_size)

    agents = [
            Actor.remote(
                actor_id=i,
                trajectory=trajectory,
                policy=ImpalaAgent(
                    model=Model(),
                    action_size=action_size,
                    device=torch.device('cpu'))
                ) for i in range(num_workers)]

    params = [learner.get_weights.remote()]
    done_id, params = ray.wait(params)
    params = ray.get(done_id)[0]
    params = convert_gpu_to_cpu(params)
    rollouts = [agent.rollout.remote(params) for agent in agents]

    while True:
        
        done, rollouts = ray.wait(rollouts)
        traj, info = ray.get(done)[0]

        learner.append.remote(
                state=traj.state, next_state=traj.next_state,
                reward=traj.reward, done=traj.done,
                action=traj.action, mu=traj.mu)

        params = [learner.get_weights.remote()]
        done_id, params = ray.wait(params)
        params = ray.get(done_id)[0]
        params = convert_gpu_to_cpu(params)
        rollouts.extend([
            agents[info['id']].rollout.remote(params)])


if __name__ == '__main__':
    ray.init(num_gpus=1)
    cpu_count = multiprocessing.cpu_count()
    cpu_count = 16
    main(num_workers=cpu_count)
