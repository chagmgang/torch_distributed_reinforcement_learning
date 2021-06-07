import gym
import torch

import numpy as np

from rl.agent.dqn import DQNAgent
from rl.model.cartpole import ValueModel
from rl.buffer.apex import Memory

def main():

    env = gym.make('CartPole-v0')
    agent = DQNAgent(
            model=ValueModel(),
            action_size=2,
            device=torch.device('cpu'))

    replay_buffer = Memory(capacity=int(1e5))
    episode = 0

    train_step = 0

    while True:

        state = env.reset()
        done = False
        score = 0
        episode += 1
        epsilon = 1 / (episode * 0.05 + 1)

        while not done:

            action = agent.get_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)

            score += reward

            td_error = agent.td_error(
                    np.stack([state]), np.stack([next_state]),
                    np.stack([action]), np.stack([reward]), np.stack([done]))

            replay_buffer.add(
                    td_error[0],
                    [state, next_state, action,
                     reward, done])

            state = next_state

            if episode > 100:

                train_step += 1

                minibatch, idxs, is_weight = replay_buffer.sample(32)
                minibatch = np.stack(minibatch)

                data_state = np.stack(minibatch[:, 0])
                data_next_state = np.stack(minibatch[:, 1])
                data_action = np.stack(minibatch[:, 2])
                data_reward = np.stack(minibatch[:, 3])
                data_done = np.stack(minibatch[:, 4])

                loss, td_error = agent.train(
                        data_state, data_next_state,
                        data_action, data_reward, data_done, is_weight)

                for i in range(len(idxs)):
                    replay_buffer.update(idxs[i], td_error[i])

                if train_step % 100 == 0:

                    agent.main_to_target()

        print(episode, score)

if __name__ == '__main__':
    main()
