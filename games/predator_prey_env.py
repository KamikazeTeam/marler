import curses
import time
import gym
import numpy as np
import argparse


def init_args(parser):
    parser.add_argument('--n_predator', type=int, default=8, help="Number of agents")
    parser.add_argument('--n_prey', type=int, default=1, help="Total number of preys in play")
    parser.add_argument('--dim', type=int, default=15, help="Dimension of box")
    parser.add_argument('--vision', type=int, default=2, help="Vision of predator")
    parser.add_argument('--n_action', type=int, default=5, help="number of actions")
    parser.add_argument('--enemy_comm', action="store_true", default=False, help="Whether prey can communicate.")
    parser.add_argument('--mode', default='mixed', type=str, help='cooperative|competitive|mixed (default: mixed)')


class PredatorPreyEnv(gym.Env):
    def close(self):
        curses.endwin()

    def __init__(self, args, render=True):
        self.n_predator = 18
        self.n_prey = 1
        self.dim = 15
        self.vision = 2
        self.n_action = 5
        self.enemy_comm = 0
        self.mode = 'cooperative'
        for key in args:
            setattr(self, key, args[key])  # getattr(args, key))
        self.OUTSIDE_CLASS = 1
        self.PREDATOR_CLASS = 3
        self.PREY_CLASS = 2
        self.TIMESTEP_PENALTY = -0.05
        self.PREY_REWARD = 0
        self.POS_PREY_REWARD = 0.05
        if render:
            self.std_scr = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_RED, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_CYAN, -1)
            curses.init_pair(4, curses.COLOR_GREEN, -1)
        # (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT, 4: STAY)
        self.dims = (self.dim, self.dim)
        self.BASE = (self.dims[0] * self.dims[1])
        self.OUTSIDE_CLASS += self.BASE
        self.PREDATOR_CLASS += self.BASE
        self.PREY_CLASS += self.BASE
        # Setting max vocab size for 1-hot encoding
        self.vocab_size = 1 + 1 + self.BASE + 1 + 1
        #          predator + prey + grid + outside
        self.episode_over = False
        self.empty_grid = np.arange(self.BASE).reshape(self.dims)
        # Mark agents in grid
        # self.empty_grid[self.predator_loc[:,0], self.predator_loc[:,1]] = self.predator_ids
        # self.empty_grid[self.prey_loc[:,0], self.prey_loc[:,1]] = self.prey_ids
        self.empty_grid = np.pad(self.empty_grid, self.vision, 'constant', constant_values=self.OUTSIDE_CLASS)
        self.empty_grid_onehot = self._onehot_initialization(self.empty_grid)
        self.stat = dict()  # stat - like success ratio
        self.predator_loc, self.prey_loc = None, None
        self.reached_prey = None
        self.metadata['n_agents'] = self.n_predator + self.n_prey - 1
        self.metadata['observation_space'] = \
            [gym.spaces.Box(low=0, high=255, shape=((2*self.vision)+1, (2*self.vision)+1, self.vocab_size),
                            dtype=np.uint8) for _ in range(self.n_predator)] + \
            [gym.spaces.Box(low=0, high=255, shape=((2*self.vision)+1, (2*self.vision)+1, self.vocab_size),
                            dtype=np.uint8) for _ in range(self.n_prey)]
        # Observation for each storage will be vision * vision ndarray
        self.observation_space = \
            gym.spaces.Box(low=0, high=255, shape=(self.metadata['n_agents'] * self.vocab_size,
                                                   (2*self.vision)+1, (2*self.vision)+1), dtype=np.uint8)
        # Actual observation will be of the shape 1 * n_predator * (2v+1) * (2v+1) * vocab_size
        self.metadata['action_space'] = \
            [gym.spaces.Discrete(self.n_action) for _ in range(self.n_predator)] + \
            [gym.spaces.Discrete(self.n_action) for _ in range(self.n_prey)]
        self.action_space = gym.spaces.Discrete(self.n_action)  # MultiDiscrete([self.n_action])
        return

    def _onehot_initialization(self, a):
        out = np.zeros(a.shape + (self.vocab_size,), dtype=int)
        grid = np.ogrid[tuple(map(slice, a.shape))]
        grid.insert(2, a)
        out[tuple(grid)] = 1
        return out

    def reset(self, *, seed=None, options=None):
        self.episode_over = False
        self.stat.clear()
        idx = np.random.choice(np.prod(self.dims), (self.n_predator + self.n_prey), replace=False)
        loc = np.vstack(np.unravel_index(idx, self.dims)).T
        self.predator_loc, self.prey_loc = loc[:self.n_predator], loc[self.n_predator:]
        self.reached_prey = np.zeros(self.n_predator)
        # Observation will be n_predator * vision * vision ndarray
        return self._get_obs(), {}  # 8,5,5,229

    def step(self, action):
        """ action : list/ndarray of length m, containing the indexes of what lever each 'm' chosen agents pulled.
            reward (float) : Ratio of Number of discrete levers pulled to total number of levers.
            episode_over (bool) : Will be true as episode length is 1
            info (dict) : diagnostic information useful for debugging. """
        if self.episode_over:
            raise RuntimeError("Episode is done")
        assert np.all(action <= self.n_action), "Actions should be in the range [0,n_action)."
        for i, a in enumerate(action):
            self._take_action(i, a)
        self.episode_over = False
        debug = {'predator_loc': self.predator_loc, 'prey_loc': self.prey_loc}
        return self._get_obs(), self._get_reward(), self.episode_over, False, debug

    def _get_obs(self):
        grid = self.empty_grid_onehot.copy()
        for i, p in enumerate(self.predator_loc):
            grid[p[0] + self.vision, p[1] + self.vision, self.PREDATOR_CLASS] += 1
        for i, p in enumerate(self.prey_loc):
            grid[p[0] + self.vision, p[1] + self.vision, self.PREY_CLASS] += 1
        _obs = []
        for p in self.predator_loc:
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            _obs.append(grid[slice_y, slice_x])
        if self.enemy_comm:
            for p in self.prey_loc:
                slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)
                slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
                _obs.append(grid[slice_y, slice_x])
        _obs = np.stack(_obs)
        _obs = np.transpose(_obs, axes=[0, 3, 1, 2])  # change order to combine multi-agent axis with channel axis
        _obs = _obs.reshape((_obs.shape[0]*_obs.shape[1], *_obs.shape[2:]))
        return _obs.astype(np.uint8)

    def _take_action(self, idx, act):
        if idx >= self.n_predator:  # prey action
            return
        if self.reached_prey[idx] == 1:
            return
        if act == 5:  # STAY action
            return
        # UP
        if act == 0 and self.empty_grid[max(0, self.predator_loc[idx][0] + self.vision - 1),
                                        self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = max(0, self.predator_loc[idx][0] - 1)
        # RIGHT
        elif act == 1 and self.empty_grid[self.predator_loc[idx][0] + self.vision,
                                          min(self.dims[1] - 1,
                                              self.predator_loc[idx][1] + self.vision + 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = min(self.dims[1] - 1, self.predator_loc[idx][1] + 1)
        # DOWN
        elif act == 2 and self.empty_grid[min(self.dims[0] - 1, self.predator_loc[idx][0] + self.vision + 1),
                                          self.predator_loc[idx][1] + self.vision] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][0] = min(self.dims[0] - 1, self.predator_loc[idx][0] + 1)
        # LEFT
        elif act == 3 and self.empty_grid[self.predator_loc[idx][0] + self.vision,
                                          max(0, self.predator_loc[idx][1] + self.vision - 1)] != self.OUTSIDE_CLASS:
            self.predator_loc[idx][1] = max(0, self.predator_loc[idx][1] - 1)

    def _get_reward(self):
        n = self.n_predator if not self.enemy_comm else self.n_predator + self.n_prey
        _reward = np.full(n, self.TIMESTEP_PENALTY)

        on_prey = np.where(np.all(self.predator_loc == self.prey_loc, axis=1))[0]
        nb_predator_on_prey = on_prey.size

        if self.mode == 'cooperative':
            _reward[on_prey] = self.POS_PREY_REWARD * nb_predator_on_prey
        elif self.mode == 'competitive':
            if nb_predator_on_prey:
                _reward[on_prey] = self.POS_PREY_REWARD / nb_predator_on_prey
        elif self.mode == 'mixed':
            _reward[on_prey] = self.PREY_REWARD
        else:
            raise RuntimeError("Incorrect mode, Available modes: [cooperative|competitive|mixed]")

        self.reached_prey[on_prey] = 1

        if np.all(self.reached_prey == 1) and self.mode == 'mixed':
            self.episode_over = True

        # Prey reward
        if nb_predator_on_prey == 0:
            _reward[self.n_predator:] = -1 * self.TIMESTEP_PENALTY
        else:
            # TODO: discuss & finalise
            _reward[self.n_predator:] = 0

        # Success ratio
        if self.mode != 'competitive':
            if nb_predator_on_prey == self.n_predator:
                self.stat['success'] = 1
            else:
                self.stat['success'] = 0
        return _reward

    def render(self, mode='human', close=False):
        grid = np.zeros(self.BASE, dtype=object).reshape(self.dims)
        self.std_scr.clear()
        for p in self.predator_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'X'
            else:
                grid[p[0]][p[1]] = 'X'
        for p in self.prey_loc:
            if grid[p[0]][p[1]] != 0:
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]) + 'P'
            else:
                grid[p[0]][p[1]] = 'P'
        _vspace, _space, _center = 2, 4, 5
        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if item != 0:
                    if 'X' in item and 'P' in item:
                        self.std_scr.addstr(row_num * _vspace, idx * _space, item.center(_center), curses.color_pair(3))
                    elif 'X' in item:
                        self.std_scr.addstr(row_num * _vspace, idx * _space, item.center(_center), curses.color_pair(1))
                    else:
                        self.std_scr.addstr(row_num * _vspace, idx * _space, item.center(_center), curses.color_pair(2))
                else:
                    self.std_scr.addstr(row_num * _vspace, idx * _space, '0'.center(_center), curses.color_pair(4))
        self.std_scr.addstr(len(grid) * _vspace, 0, '\n')
        self.std_scr.refresh()
        time.sleep(.31)


def main():
    parser = argparse.ArgumentParser()
    init_args(parser)
    args = parser.parse_args()
    env = PredatorPreyEnv(args.__dict__)
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
