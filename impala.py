import gym
import torch

from rl.model.cartpole import Model
from rl.buffer.impala import ImpalaBuffer
from rl.agent.impala import ImpalaAgent
import numpy as np

def main():

    env = gym.make('CartPole-v0')
    buffers = ImpalaBuffer(
            trajectory=20, buffer_size=128)

    agent = ImpalaAgent(
            model=Model(),
            action_size=2,
            device=torch.device('cpu'))

    state = env.reset()
    done = False
    score = 0
    episode = 0

    while True:

        action, mu = agent.get_action(state)

        next_state, reward, done, _ = env.step(action)

        score += reward
        buffers.append(
                state, next_state,
                action, reward,
                done, mu)

        state = next_state

        if done:
            print(episode, score)
            state = env.reset()
            score = 0
            episode += 1

        if len(buffers) > 32:
            sample = buffers.sample(16)
            agent.train(
                    state=sample.state,
                    next_state=sample.next_state,
                    reward=sample.reward,
                    done=sample.done,
                    action=sample.action,
                    mu=sample.mu)

if __name__ == '__main__':
    main()
