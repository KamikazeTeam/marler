import numpy as np
import os
import sys
import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
from itertools import cycle


atari_names = ["BeamRiderNoFrameskip-v4", "BreakoutNoFrameskip-v4", "PongNoFrameskip-v4", "QbertNoFrameskip-v4"]
color_list = ['red', 'orange', 'green', 'cyan', 'blue', 'purple']  # ,'black']
marker_list = ['+', '.', 'o', '*']
# ",",".","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","d","D","|","_","","","","",""


def fig_all_curves(_file_name, _curve_length, _suffix, _suffix_list, _file_id):
    colors = cycle(color_list)
    markers = cycle(marker_list)
    print(_suffix)
    for i in range(_file_id % len(color_list)):
        print(i)
        next(colors)
    for i in range(abs(hash(_suffix)) % len(marker_list)):
        next(markers)
    lines = open(_file_name, 'r').read().splitlines()
    for i, line in enumerate(lines):
        elements = line.split('|')
        heads = elements[0].replace('/', ':').split(':')
        print(heads)
        title = heads[1].split('_')[0]  # .split('-')[0]
        env = heads[1].split('_')[1:]
        storage = heads[2].split('_')
        model = heads[3].split('_')
        algorithm = heads[4].split('_')
        approximate_function = heads[5].split('_')
        start_step = 0
        labels = '_'.join(approximate_function)

        total_steps = _curve_length * 1000000
        score_min, score_max = -50, 50
        if title == 'RoboschoolAnt-v1':
            score_min, score_max = -0, 3000
        if title == 'RoboschoolHalfCheetah-v1':
            score_min, score_max = -0, 3500
        if title == 'RoboschoolHopper-v1':
            score_min, score_max = -0, 2500
        if title == 'RoboschoolReacher-v1':
            score_min, score_max = -50, 25
        if title == 'RoboschoolWalker2d-v1':
            score_min, score_max = -0, 1500

        if title == 'BeamRiderNoFrameskip-v4':
            score_min, score_max = -0, 8000
        if title == 'BreakoutNoFrameskip-v4':
            score_min, score_max = -0, 600  # 1000
        if title == 'PongNoFrameskip-v4':
            score_min, score_max = -25, 25
        if title == 'QbertNoFrameskip-v4':
            score_min, score_max = -0, 20000

        if title == 'LunarLander-v2':
            score_min, score_max = -200, 300
        if title == 'LunarLanderContinuous-v2':
            score_min, score_max = -200, 300
        if title == 'BipedalWalker-v2':
            score_min, score_max = -200, 300
        if title == 'BipedalWalkerHardcore-v2':
            score_min, score_max = -200, 300

        if title == 'MountainCar-v0':
            score_min, score_max = -250, -100
        if title == 'MountainCarContinuous-v0':
            score_min, score_max = -100, 100
        if title == 'Pendulum-v0':
            score_min, score_max = -2100, -100
        if title == 'CartPole-v1':
            score_min, score_max = -100, 600
        if title == 'Acrobot-v1':
            score_min, score_max = -700, 0

        xs, ys, yv = [], [], []
        for element in elements[1].split(',')[:-1]:
            xs.append(int(element) + start_step)
        for element in elements[2].split(',')[:-1]:
            ys.append(float(element))
        for element in elements[3].split(',')[:-1]:
            yv.append(float(element))
        xs = np.array(xs)
        ys = np.array(ys)
        yv = np.array(yv)

        plt.figure(title, figsize=(16, 9))
        color = next(colors)
        marker = next(markers)
        markers_on = [-1]
        start = 0  # 3
        plt.plot(xs[start:], ys[start:], color=color, alpha=1.0, linewidth=0.5, marker=marker, markersize=3,
                 markevery=markers_on, label=labels)
        plt.fill_between(xs[start:], ys[start:] - yv[start:], ys[start:] + yv[start:], facecolor=color, alpha=0.3)
        # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', borderaxespad=0)
        plt.legend(loc='lower left')
        axes = plt.gca()
        axes.set_xticks(np.arange(0, total_steps, total_steps / 10))
        axes.set_yticks(np.arange(score_min, score_max, (score_max - score_min) / 10))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.xlim([0, total_steps])
        plt.ylim([score_min, score_max])
        plt.tick_params(labelsize=8)
        plt.title(title)
        plt.xlabel('Number of Time Steps')
        plt.ylabel('Episodic Reward')
        plt.grid(linewidth=0.1)
        plt.savefig('compare_' + ','.join(_suffix_list) + '_' + title + '_' + str(_curve_length) + '.png',
                    dpi=300, facecolor="azure", bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        curve_length = int(sys.argv[1])
    else:
        curve_length = 10  # M
    if len(sys.argv) > 2:
        suffix_list = str(sys.argv[2]).split(',')
    else:
        suffix_list = ['']
    files = os.listdir('./')
    files.sort()
    for _, file_name in enumerate(files):
        if os.path.isdir(file_name):
            continue
        if file_name[:7] != 'results' or file_name[-4:] == '.png':
            continue
        suffix = file_name.split('_')[0][7:]
        if suffix in suffix_list:
            file_id = suffix_list.index(suffix)
            fig_all_curves(file_name, curve_length, suffix, suffix_list, file_id)
