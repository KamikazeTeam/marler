import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt


def init_args(parser):
    parser.add_argument('--n', type=int, default=1000, help="Number of agents")
    parser.add_argument('--t', type=int, default=80, help="Number of steps")
    # parser.add_argument('--enemy_comm', action="store_true", default=False, help="Whether prey can communicate.")
    # parser.add_argument('--mode', default='mixed', type=str, help='cooperative|competitive|mixed (default: mixed)')


class TalentLuck(gym.Env):
    def __init__(self, args, render=True):
        # for key in args:
        #     setattr(self, key, args[key])
        self.n = args['n']
        self.t = args['t']
        self.asset = np.zeros((self.n, self.t+1))
        self.talent = np.random.normal(0.6, 0.1, self.n)
        self.talent = np.clip(self.talent, 0, 1)
        # generate events
        self.event = np.random.uniform(0, 1, (self.n, self.t+1))
        event_good = (self.event >= 0.97).astype(np.int64)
        event_bad = -(self.event <= 0.03).astype(np.int64)
        self.event = event_good + event_bad

        # running simulation
        for i in range(self.n):
            self.asset[i][0] = 1.0
            for j in range(self.t):
                if self.event[i][j] == -1:
                    self.asset[i][j+1] = self.asset[i][j] * 0.5
                elif self.event[i][j] == 1:
                    if np.random.uniform(0, 1, 1) < self.talent[i]:
                        self.asset[i][j+1] = self.asset[i][j] * 2.0
                    else:
                        self.asset[i][j+1] = self.asset[i][j]
                else:
                    self.asset[i][j+1] = self.asset[i][j]
        final = self.asset[:, -1]

        # show distribution of events
        event_good_num = np.sum(event_good, axis=-1)
        # print(event_good_num)
        fig = plt.figure()
        count, bins, ignored = plt.hist(event_good_num, np.max(event_good_num)-np.min(event_good_num), density=False)
        axes = plt.gca()
        axes.set_yscale('log')
        # axes.set_xscale('log')
        plt.grid(linewidth=0.1)
        # plt.show()
        # exit()

        # show distribution of final scores
        fig = plt.figure()
        count, bins, ignored = plt.hist(final, 100, density=False)
        axes = plt.gca()
        axes.set_yscale('log')
        # axes.set_xscale('log')
        plt.grid(linewidth=0.1)
        fig = plt.figure()
        plt.plot(bins[1:], count, color='red', alpha=1.0, linewidth=0.3, marker='.')
        axes = plt.gca()
        axes.set_yscale('log')
        axes.set_xscale('log')
        plt.grid(linewidth=0.1)

        plt.show()
        exit()
        # self.metadata['n_agents'] = self.n_predator + self.n_prey - 1
        # self.metadata['observation_space'] = \
        #     [gym.spaces.Box(low=0, high=255, shape=((2*self.vision)+1, (2*self.vision)+1, self.vocab_size),
        #                     dtype=np.uint8) for _ in range(self.n_predator)] + \
        #     [gym.spaces.Box(low=0, high=255, shape=((2*self.vision)+1, (2*self.vision)+1, self.vocab_size),
        #                     dtype=np.uint8) for _ in range(self.n_prey)]
        # # Observation for each storage will be vision * vision ndarray
        # self.observation_space = \
        #     gym.spaces.Box(low=0, high=255, shape=(self.metadata['n_agents'] * self.vocab_size,
        #                                            (2*self.vision)+1, (2*self.vision)+1), dtype=np.uint8)
        # # Actual observation will be of the shape 1 * n_predator * (2v+1) * (2v+1) * vocab_size
        # self.metadata['action_space'] = \
        #     [gym.spaces.Discrete(self.n_action) for _ in range(self.n_predator)] + \
        #     [gym.spaces.Discrete(self.n_action) for _ in range(self.n_prey)]
        # self.action_space = gym.spaces.Discrete(self.n_action)

        # count, bins, ignored = plt.hist(self.talent, 30, density=True)
        # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
        #          linewidth=2, color='r')
        # plt.show()
        return

    def reset(self, *, seed=None, options=None):
        return self._get_obs(), {}

    def step(self, action):
        return self._get_obs(), self._get_reward(), self.episode_over, False, debug

    def render(self, mode='human', close=False):
        pass


def main():
    parser = argparse.ArgumentParser()
    init_args(parser)
    args = parser.parse_args()
    env = TalentLuck(args.__dict__)
    episodes = 0
    try:
        while episodes < 50:
            obs, info = env.reset()
            done = False
            while not done:
                actions = []
                for _ in range(env.metadata['n_agents']):
                    actions.append(env.action_space.sample())
                actions = np.array(actions)
                actions = actions.squeeze()
                obs, reward, done, _, info = env.step(actions)
                env.render()
            episodes += 1
        env.close()
    except KeyboardInterrupt:
        env.close()


if __name__ == '__main__':
    main()
