import ray
import gym
import time
import torch

from rl.model.cartpole import Model
from rl.buffer.impala import LocalBuffer
from rl.buffer.impala import GlobalBuffer
from rl.agent.impala import ImpalaAgent
from threading import Thread
from tensorboardX import SummaryWriter

import numpy as np


@ray.remote
class Actor(object):

    def __init__(self, actor_id, trajectory):

        self.env = gym.make('CartPole-v0')
        
        self.id = actor_id
        self.trajectory = trajectory

        self.policy = ImpalaAgent(
                model=Model(),
                action_size=2,
                device=torch.device('cpu'))

        self.local_buffer = LocalBuffer(self.trajectory)
        self.writer = SummaryWriter(f'runs/{self.id}')
        self.running = Thread(
                target=self.runner)
        self.running.start()

    def set_weights(self, param):
        self.policy.set_weights(param)
        if len(self.local_buffer) == self.trajectory:
            data = self.local_buffer.sample()
        else:
            data = None
        info = {'id': self.id,
                'data': data}
        self.local_buffer = LocalBuffer(self.trajectory)
        return None, info

    def runner(self):

        state = self.env.reset()
        done = False
        score = 0
        episode = 0

        while True:

            action, mu = self.policy.get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            score += reward
            self.local_buffer.append(
                    state, next_state, action,
                    reward, done, mu)

            state = next_state

            if done:
                self.writer.add_scalar('data/score', score, episode)
                print(f'actor id : {self.id} | episode : {episode} | score : {score}')
                state = self.env.reset()
                score = 0
                episode += 1

def main(num_workers):

    trajectory = 20
    batch_size = 16
    buffer_size = 128

    writer = SummaryWriter('runs/learner')

    global_buffer = GlobalBuffer(
            trajectory=trajectory,
            buffer_size=buffer_size)
    learner = ImpalaAgent(
            model=Model(), action_size=2,
            device=torch.device('cpu'))

    train_step = 0

    agents = [
            Actor.remote(actor_id=i, trajectory=trajectory)
            for i in range(num_workers)]

    params = learner.get_weights()
    samples = [agent.set_weights.remote(params) for agent in agents]

    while True:

        done_id, samples = ray.wait(samples)
        sample, info = ray.get(done_id)[0]
        if info['data'] is not None:

            data = info['data']
            global_buffer.append(
                    state=data.state,
                    next_state=data.next_state,
                    action=data.action,
                    reward=data.reward,
                    done=data.done,
                    mu=data.mu)

        if len(global_buffer) > 2 * batch_size:
            train_data = global_buffer.sample(batch_size)
            p_loss, v_loss, ent = learner.train(
                    state=train_data.state, next_state=train_data.next_state,
                    reward=train_data.reward, done=train_data.done,
                    action=train_data.action, mu=train_data.mu)
            train_step += 1
            writer.add_scalar('data/pi_loss', float(p_loss), train_step)
            writer.add_scalar('data/value_loss', float(v_loss), train_step)
            writer.add_scalar('data/ent', float(ent), train_step)

        params = learner.get_weights()
        samples.extend([
            agents[info['id']].set_weights.remote(params)])

if __name__ == '__main__':
    ray.init()
    main(num_workers=4)
