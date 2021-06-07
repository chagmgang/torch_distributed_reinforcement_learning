import ray

from rl.buffer.apex import LocalBuffer

@ray.remote
class Actor(object):

    def __init__(self, actor_id):

        self.env = gym.make('CartPole-v0')

def main(num_workers):

    print(num_workers)

if __name__ == '__main__':
    ray.init()
    main(num_workers=1)
