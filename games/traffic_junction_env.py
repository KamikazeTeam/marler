import math
import curses
import time
import gym
from traffic_helper import *
import argparse
import cv2
from mss import mss


def init_args(parser):
    parser.add_argument('--n_car', type=int, default=16, help="max number of cars")
    parser.add_argument('--difficulty', type=str, default='hard', help="Difficulty level, easy|medium|hard")
    parser.add_argument('--dim', type=int, default=12, help="Dimension of box (i.e length of road) ")
    parser.add_argument('--vision', type=int, default=1, help="Vision of car")
    parser.add_argument('--add_rate_min', type=float, default=0.05, help="rate to add car(till curr_start)")
    parser.add_argument('--add_rate_max', type=float, default=0.2, help=" max rate at which to add car")
    parser.add_argument('--curr_start', type=float, default=0, help="start making harder after this many epochs")
    parser.add_argument('--curr_end', type=float, default=0, help="when to make the game hardest")
    parser.add_argument('--vocab_type', type=str, default='bool', help="Type of location vector, bool|scalar")


class TrafficJunctionEnv(gym.Env):
    def close(self):
        self.vWriter.release()
        curses.endwin()

    def __init__(self, args, render):
        self.n_car = 16
        self.difficulty = 'hard'
        self.dim = 12
        self.vision = 1
        self.exact_rate = self.add_rate = self.add_rate_min = 0.05
        self.add_rate_max = 0.2
        self.curr_start, self.curr_end, self.epoch_last_update = 0, 0, 0
        self.vocab_type = 'bool'  # Setting max vocab size for 1-hot encoding
        for key in args:
            setattr(self, key, getattr(args, key))
        self.OUTSIDE_CLASS = 0
        self.ROAD_CLASS = 1
        self.CAR_CLASS = 2
        self.TIMESTEP_PENALTY = -0.01
        self.CRASH_PENALTY = -10
        if render:
            self.bounding_box = {'top': 100, 'left': 100, 'width': 800, 'height': 600}
            self.screenshot = mss()
            video_name = 'video_out_put.mp4'
            fps, fourcc = 20, cv2.VideoWriter_fourcc(*'mp4v')  # 'M','J','P','G')
            self.vWriter = cv2.VideoWriter(video_name, fourcc, fps, (800, 600))
            self.std_scr = curses.initscr()
            curses.start_color()
            curses.use_default_colors()
            background = -1  # curses.COLOR_MAGENTA  # COLOR_BLACK, COLOR_WHITE
            curses.init_pair(1, curses.COLOR_RED, background)
            curses.init_pair(2, curses.COLOR_YELLOW, background)
            curses.init_pair(3, curses.COLOR_CYAN, background)
            curses.init_pair(4, curses.COLOR_GREEN, background)
            curses.init_pair(5, curses.COLOR_BLUE, background)
        self.dims = (self.dim, self.dim)
        if self.difficulty == 'easy':
            assert self.dims[0] % 2 == 1  # no. of dims should odd for easy case.
            assert self.dims[0] >= 4 + self.vision, 'Min dim: 4 + vision'
        if self.difficulty == 'medium':
            assert self.dims[0] % 2 == 0, 'Only even dimension supported for now.'
            assert self.dims[0] >= 4 + self.vision, 'Min dim: 4 + vision'
        if self.difficulty == 'hard':
            assert self.dims[0] % 3 == 0, 'Hard version works for multiple of 3. dim. only.'
            assert self.dims[0] >= 9, 'Min dim: 9'
        self.n_action = 2  # Define what a storage can do - # (0: GAS, 1: BRAKE) i.e. (0: Move 1-step, 1: STAY)
        self.action_space = gym.spaces.Discrete(self.n_action)
        n_road = {'easy': 2, 'medium': 4, 'hard': 8}
        self.n_path = math.factorial(n_road[self.difficulty]) // math.factorial(n_road[self.difficulty] - 2)
        if self.vocab_type == 'bool':
            dim_sum = self.dims[0] + self.dims[1]
            base = {'easy': dim_sum, 'medium': 2 * dim_sum, 'hard': 4 * dim_sum}
            self.BASE = base[self.difficulty]
            self.OUTSIDE_CLASS += self.BASE
            self.CAR_CLASS += self.BASE
            self.vocab_size = 1 + self.BASE + 1 + 1  # car_type + base + outside + 0-index
            self.observation_space = gym.spaces.Tuple((
                gym.spaces.Discrete(self.n_action),
                gym.spaces.Discrete(self.n_path),
                gym.spaces.MultiBinary((2 * self.vision + 1, 2 * self.vision + 1, self.vocab_size))))
        else:
            self.vocab_size = 1 + 1  # r_i, (x,y), vocab = [road class + car]
            # Observation for each storage will be 4-tuple of (last_act, r_i, len(dims), vision * vision * vocab)
            self.observation_space = gym.spaces.Tuple((
                gym.spaces.Discrete(self.n_action),
                gym.spaces.Discrete(self.n_path),
                gym.spaces.MultiDiscrete(list(self.dims)),  # changed tuple to list # by SY
                gym.spaces.MultiBinary((2 * self.vision + 1, 2 * self.vision + 1, self.vocab_size))))
            # Actual observation will be of the shape 1 * n_car * ((x,y) , (2v+1) * (2v+1) * vocab_size)
        self.empty_grid = np.full(self.dims[0] * self.dims[1], self.OUTSIDE_CLASS, dtype=int).reshape(self.dims)
        w, h = self.dims  # Mark the roads
        roads = get_road_blocks(w, h, self.difficulty)
        for road in roads:
            self.empty_grid[road] = self.ROAD_CLASS
        if self.vocab_type == 'bool':
            self.route_grid = self.empty_grid.copy()
            start = 0
            for road in roads:
                sz = int(np.prod(self.empty_grid[road].shape))
                self.empty_grid[road] = np.arange(start, start + sz).reshape(self.empty_grid[road].shape)
                start += sz
        self.empty_grid = np.pad(self.empty_grid, self.vision, 'constant', constant_values=self.OUTSIDE_CLASS)
        self.empty_grid_onehot = self._onehot_initialization(self.empty_grid)
        if self.difficulty == 'easy':
            h, w = self.dims
            self.routes = {'TOP': [], 'LEFT': []}
            full = [(i, w // 2) for i in range(h)]  # 0 refers to UP to DOWN, type 0
            self.routes['TOP'].append(np.array([*full]))
            full = [(h // 2, i) for i in range(w)]  # 1 refers to LEFT to RIGHT, type 0
            self.routes['LEFT'].append(np.array([*full]))
            self.routes = list(self.routes.values())
        else:
            route_grid = self.route_grid if self.vocab_type == 'bool' else self.empty_grid
            self.routes = get_routes(self.dims, route_grid, self.difficulty)
            paths = []  # Convert/unroll routes which is a list of paths' list
            for r in self.routes:
                for p in r:
                    paths.append(p)
            assert len(paths) == self.n_path  # Check number of paths
            assert self._unittest_path(paths)  # Test all paths
        self.episode_over, self.has_failed = False, 0
        self.alive_mask, self.wait, self.cars_in_sys = None, None, None
        self.chosen_path, self.route_id = None, None
        self.car_ids, self.car_loc, self.car_last_act, self.car_route_loc = None, None, None, None
        self.stat = None
        self.is_completed = None
        self.n_agents = self.n_car
        return

    @staticmethod
    def _unittest_path(paths):
        for i, p in enumerate(paths[:-1]):
            next_dif = p - np.row_stack([p[1:], p[-1]])
            next_dif = np.abs(next_dif[:-1])
            step_jump = np.sum(next_dif, axis=1)
            if np.any(step_jump != 1):
                print("Any", p, i)
                return False
            if not np.all(step_jump == 1):
                print("All", p, i)
                return False
        return True

    def _onehot_initialization(self, a):
        if self.vocab_type == 'bool':
            n_cols = self.vocab_size
        else:
            n_cols = self.vocab_size + 1  # 1 is for outside class which will be removed later.
        out = np.zeros(a.shape + (n_cols,), dtype=int)
        grid = np.ogrid[tuple(map(slice, a.shape))]
        grid.insert(2, a)
        out[tuple(grid)] = 1
        return out

    def reset(self, *, seed=None, options=None):  # self, epoch=None):
        self.episode_over, self.has_failed = False, 0
        self.alive_mask, self.wait, self.cars_in_sys = np.zeros(self.n_car), np.zeros(self.n_car), 0
        self.chosen_path = [0] * self.n_car  # Chosen path for each car
        self.route_id = [-1] * self.n_car  # when dead => no route, must be masked by trainer.
        self.car_ids = np.arange(self.CAR_CLASS, self.CAR_CLASS + self.n_car)  # Current car to enter system
        # Starting loc of car: a place where everything is outside class
        self.car_loc = np.zeros((self.n_car, len(self.dims)), dtype=int)
        self.car_last_act = np.zeros(self.n_car, dtype=int)  # last act GAS when awake
        self.car_route_loc = np.full(self.n_car, -1)
        self.stat = dict()  # stat - like success ratio
        # # set add rate according to the curriculum
        # epoch_range = (self.curr_end - self.curr_start)
        # add_rate_range = (self.add_rate_max - self.add_rate_min)
        # if epoch is not None and epoch_range > 0 and add_rate_range > 0 and epoch > self.epoch_last_update:
        #     self.curriculum(epoch)
        #     self.epoch_last_update = epoch
        # Observation will be n_car * vision * vision ndarray
        return self._get_obs(), {}

    # def curriculum(self, epoch):
    #     step_size = 0.01
    #     step = (self.add_rate_max - self.add_rate_min) / (self.curr_end - self.curr_start)
    #     if self.curr_start <= epoch < self.curr_end:
    #         self.exact_rate = self.exact_rate + step
    #         self.add_rate = step_size * (self.exact_rate // step_size)

    def step(self, action):
        """ action : shape - either n_car or n_car x 1
            reward (n_car x 1) : PENALTY for each timestep when in sys & CRASH PENALTY on crashes.
            episode_over (bool) : Will be true when episode gets over."""
        if self.episode_over:
            raise RuntimeError("Episode is done")
        assert np.all(action <= self.n_action), "Actions should be in the range [0,n_action)."
        assert len(action) == self.n_car, "Action for each storage should be provided."
        self.is_completed = np.zeros(self.n_car)  # No one is completed before taking action
        for i, a in enumerate(action):
            self._take_action(i, a)
        self._add_cars()
        debug = {'alive_mask': np.copy(self.alive_mask),
                 'wait': self.wait,
                 'cars_in_sys': self.cars_in_sys,
                 'car_loc': self.car_loc,
                 'is_completed': np.copy(self.is_completed)}
        self.stat['success'] = 1 - self.has_failed
        self.stat['add_rate'] = self.add_rate
        return self._get_obs(), self._get_reward(), self.episode_over, False, debug

    def _get_obs(self):
        h, w = self.dims
        grid = self.empty_grid_onehot.copy()
        for i, p in enumerate(self.car_loc):  # Mark cars' location in Bool grid
            grid[p[0] + self.vision, p[1] + self.vision, self.CAR_CLASS] += 1
        if self.vocab_type == 'scalar':  # remove the outside class.
            grid = grid[:, :, 1:]
        _obs = []
        for i, p in enumerate(self.car_loc):
            act = self.car_last_act[i] / (self.n_action - 1)  # most recent action
            r_i = self.route_id[i] / (self.n_path - 1)  # route id
            p_norm = p / (h - 1, w - 1)  # loc
            slice_y = slice(p[0], p[0] + (2 * self.vision) + 1)  # vision square
            slice_x = slice(p[1], p[1] + (2 * self.vision) + 1)
            v_sq = grid[slice_y, slice_x]
            # when dead, all obs are 0. But should be masked by trainer.
            if self.alive_mask[i] == 0:
                act = np.zeros_like(act)
                r_i = np.zeros_like(r_i)
                p_norm = np.zeros_like(p_norm)
                v_sq = np.zeros_like(v_sq)
            if self.vocab_type == 'bool':
                o = tuple((act, r_i, v_sq))
            else:
                o = tuple((act, r_i, p_norm, v_sq))
            _obs.append(o)
        return tuple(_obs)

    def _add_cars(self):
        for r_i, routes in enumerate(self.routes):
            if self.cars_in_sys >= self.n_car:
                return
            if np.random.uniform() <= self.add_rate:  # Add car to system and set on path
                car_idx = np.arange(len(self.alive_mask))  # chose dead car on random
                idx = np.random.choice(car_idx[self.alive_mask == 0])
                self.alive_mask[idx] = 1  # make it alive
                p_i = np.random.choice(len(routes))  # choose path randomly & set it
                # make sure all self.routes have equal len/ same no. of routes
                self.route_id[idx] = p_i + r_i * len(routes)
                self.chosen_path[idx] = routes[p_i]
                self.car_route_loc[idx] = 0  # set its start loc
                self.car_loc[idx] = routes[p_i][0]
                self.cars_in_sys += 1  # increase count

    def _take_action(self, idx, act):
        if self.alive_mask[idx] == 0:  # non-active car
            return
        self.wait[idx] += 1  # add wait time for active cars
        if act == 1:  # action BRAKE such as STAY
            self.car_last_act[idx] = 1
            return
        if act == 0:  # GAS or move
            self.car_last_act[idx] = 0  # Change last act for color
            prev = self.car_route_loc[idx]
            self.car_route_loc[idx] += 1
            curr = self.car_route_loc[idx]
            # car/storage has reached end of its path
            if curr == len(self.chosen_path[idx]):
                self.alive_mask[idx] = 0
                self.wait[idx] = 0
                self.cars_in_sys -= 1
                self.car_loc[idx] = np.zeros(len(self.dims), dtype=int)  # put it at dead loc
                self.is_completed[idx] = 1
                return
            elif curr > len(self.chosen_path[idx]):
                print(curr)
                raise RuntimeError("Out of bound car path")
            prev = self.chosen_path[idx][prev]
            curr = self.chosen_path[idx][curr]
            # assert abs(curr[0] - prev[0]) + abs(curr[1] - prev[1]) == 1 or curr_path = 0
            self.car_loc[idx] = curr

    def _get_reward(self):
        _reward = np.full(self.n_car, self.TIMESTEP_PENALTY) * self.wait
        for i, l in enumerate(self.car_loc):
            if (len(np.where(np.all(self.car_loc[:i] == l, axis=1))[0]) or len(
                    np.where(np.all(self.car_loc[i + 1:] == l, axis=1))[0])) and l.any():
                _reward[i] += self.CRASH_PENALTY
                self.has_failed = 1
        return self.alive_mask * _reward

    def render(self, mode='human', close=False):
        symbol_road, symbol_car, symbol_break = '--', '[]', '[b]'
        grid = self.route_grid.copy().astype(object)
        grid[grid != self.OUTSIDE_CLASS] = symbol_road
        grid[grid == self.OUTSIDE_CLASS] = ''
        self.std_scr.clear()
        for i, p in enumerate(self.car_loc):
            if self.car_last_act[i] == 0:  # GAS
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace(symbol_road, '') + symbol_car
            else:  # BRAKE
                grid[p[0]][p[1]] = str(grid[p[0]][p[1]]).replace(symbol_road, '') + symbol_break
        _vspace, _space, _center = 2, 4, 4
        for row_num, row in enumerate(grid):
            for idx, item in enumerate(row):
                if row_num == idx == 0:  # skip top left item
                    continue
                if item != symbol_road:
                    if symbol_car in item and len(item) > 3:  # CRASH, one car accelerates
                        self.std_scr.addstr(row_num, idx*_space,
                                            item.replace('b', '').center(_center), curses.color_pair(2))
                    elif symbol_car in item:  # GAS
                        self.std_scr.addstr(row_num, idx*_space,
                                            item.replace('b', '').center(_center), curses.color_pair(5))
                    elif 'b' in item and len(item) > 3:  # CRASH
                        self.std_scr.addstr(row_num, idx*_space,
                                            item.replace('b', '').center(_center), curses.color_pair(2))
                    elif 'b' in item:
                        self.std_scr.addstr(row_num, idx*_space,
                                            item.replace('b', '').center(_center), curses.color_pair(1))
                    else:
                        self.std_scr.addstr(row_num, idx*_space,
                                            item.center(_center), curses.color_pair(2))
                else:
                    self.std_scr.addstr(row_num, idx * _space,
                                        symbol_road.center(_center), curses.color_pair(4))
        self.std_scr.addstr(len(grid), 0, '\n')
        self.std_scr.refresh()
        time.sleep(.1)
        frame = self.screenshot.grab(self.bounding_box)
        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # without this video will be error
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # change color back...
        self.vWriter.write(frame)


def main():
    parser = argparse.ArgumentParser()
    init_args(parser)
    args = parser.parse_args()
    env = TrafficJunctionEnv(args)
    episodes = 0
    try:
        while episodes < 5:
            obs, info = env.reset()
            done = False
            while not done:
                actions = []
                for _ in range(env.n_agents):
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
