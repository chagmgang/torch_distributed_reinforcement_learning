import gym
import torch

from rl.agent.impala import ImpalaAgent
from rl.model.cartpole import Model

def main():

    env = gym.make('CartPole-v0')
    agent = ImpalaAgent(
            model=Model(),
            action_size=2,
            device=torch.device('cpu'))

    state = env.reset()
    done = False
    score = 0
    episode = 0

    while True:

        states = []
        next_states = []
        actions = []
        rewards = []
        dones = []
        mus = []

        for _ in range(32):
            
            action, mu = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            score += reward
            r = 0
            if done:
                if score == 200:
                    r = 1
                else:
                    r = -1

            states.append(state)
            next_states.append(next_state)
            rewards.append(r)
            dones.append(done)
            actions.append(action)
            mus.append(mu)

            state = next_state

            if done:
                print(episode, score)
                state = env.reset()
                done = False
                score = 0
                episode += 1
                

        agent.train(
                [states], [next_states],
                [rewards], [dones], [actions], [mus])

if __name__ == '__main__':
    main()
