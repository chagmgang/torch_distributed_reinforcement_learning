import gym
import torch

from rl.agent.a2c import A2CAgent
from rl.model.cartpole import Model

def main():

    env = gym.make('CartPole-v0')
    agent = A2CAgent(
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

        for _ in range(32):
            
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            score += reward

            states.append(state)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            actions.append(action)

            state = next_state

            if done:
                print(episode, score)
                state = env.reset()
                done = False
                score = 0
                episode += 1
                

        agent.train(
                states, next_states,
                rewards, dones, actions)

if __name__ == '__main__':
    main()
