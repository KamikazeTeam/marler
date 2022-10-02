import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
from itertools import cycle


def fig_curves(folder_exp, _args_draw):
    real_data_alpha, real_data_width = 0.1, 0.1
    real_data_mean_alpha, real_data_mean_width = 0.3, 0.1
    regular_data_alpha, regular_data_width = 0.8, 0.3
    # if _args.env_mode == 'supervise':
    #     _args.env_nums, _args_draw.avg_num, realdata_alpha, realdata_width = 1, 1, 1.0, 0.5
    names_result = os.listdir(folder_exp + _args_draw.folder_sub)
    _args_draw.env_nums = len(names_result)
    filename_common = folder_exp + _args_draw.folder_sub + _args_draw.prefix
    print(filename_common, ':filename')
    lines = []
    for i in range(_args_draw.env_nums):
        file_i_lines = open(filename_common + str(i), "r").read().splitlines()
        lines.append([line for line in file_i_lines if len(line) != 0])
    num_of_exp = len(lines[0])

    color_list = cycle(['red', 'orange', 'green', 'cyan', 'blue', 'purple'])  # ,'black'])
    plt.figure(figsize=(16, 9))
    regular_x_means, regular_y_means = [], []
    for j in range(num_of_exp):
        color = next(color_list)
        xy_tuples = []
        for i in range(_args_draw.env_nums):
            records = []
            try:
                records = lines[i][j].split("|")[:-1]  # [:int(_args.max_episodes)]
            except:
                print(i, j)
                exit()
            x, y = [], []
            for record in records:  # parse every records
                xe, ye = int(record.split(',')[0]), float(record.split(',')[1])
                x.append(xe)
                y.append(ye)
                xy_tuples.append((xe, ye))
            x_mean = [x[k] for k in range(len(x))]  # -_args_draw.avg_num+1)]
            y_mean = [np.mean(y[max(0, k + 1 - _args_draw.avg_num):k + 1]) for k in range(len(y))]
            # [np.mean(y[l:l+_args_draw.avg_num]) for l in range(len(y))]#-_args_draw.avg_num+1)]
            plt.plot(x, y, color=color, alpha=real_data_alpha, linewidth=real_data_width)
            plt.plot(x_mean, y_mean, color=color, alpha=real_data_mean_alpha, linewidth=real_data_mean_width)

        # draw all envs results in one line
        if _args_draw.prefix == 'test_':
            print(xy_tuples)
        sorted_xy_tuples = sorted(xy_tuples)
        sorted_xy_tuples_x = [xy_tuple[0] for xy_tuple in sorted_xy_tuples]
        sorted_xy_tuples_y = [xy_tuple[1] for xy_tuple in sorted_xy_tuples]
        if _args_draw.prefix == 'test_':
            print(np.array(sorted_xy_tuples_y).mean())
        regular_x, regular_y = [], []
        step_start = 0  # int(_args.start_step)
        step_end = int(step_start + _args_draw.max_steps)
        step_interval = int(_args_draw.max_steps / 200)  # hard code
        for step_i in range(step_start, step_end, step_interval):
            index = next((index for index, value in enumerate(sorted_xy_tuples_x) if value > step_i),
                         len(sorted_xy_tuples_x) - 1)
            if index != 0:  # continue
                regular_x.append(step_i)
                regular_y.append(np.mean(sorted_xy_tuples_y[max(0, index - 100):index]))  # hard code
            else:
                regular_x.append(step_i)
                regular_y.append(0.0)
        plt.plot(regular_x, regular_y, color=color, alpha=regular_data_alpha, linewidth=regular_data_width)
        regular_x_means = regular_x
        regular_y_means.append(regular_y)
    # draw all exp results in one line
    regular_y_means_mean = np.array(regular_y_means).mean(axis=0)
    regular_y_means_var = np.array(regular_y_means).std(axis=0)
    plt.plot(regular_x_means, regular_y_means_mean, color='black', alpha=1.0, linewidth=0.3)
    plt.fill_between(regular_x_means, regular_y_means_mean - regular_y_means_var,
                     regular_y_means_mean + regular_y_means_var, facecolor='black', alpha=0.5)

    step_start = 0  # int(_args.start_step)
    step_end = int(step_start + _args_draw.max_steps)
    max_steps = step_end  # int(_args.max_steps)*int(_args.roll_num)*int(_args.env_num)
    axes = plt.gca()
    axes.set_xticks(np.arange(0, max_steps, max_steps / 10))  # hard code
    plt.xlim([0, max_steps])
    if _args_draw.minmax != '':
        max_score, min_score, n = float(_args_draw.minmax.split(',')[1]), float(_args_draw.minmax.split(',')[0]), 10
        if 1:  # max_score > 0 or min_score < 0:
            diff_score = (max_score - min_score) / n
            axes.set_yticks(np.arange(min_score - diff_score, max_score + diff_score, diff_score))
            plt.ylim([min_score - diff_score, max_score + diff_score])
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.tick_params(labelsize=8)
    plt.grid(linewidth=0.1)

    fig_name = folder_exp[:-1] + _args_draw.folder_sub[:-1] + _args_draw.prefix  # +str(_args.num_of_paras)
    plt.savefig(fig_name + '.png', dpi=200, facecolor="azure", bbox_inches='tight')  # pad_inches=0)
    plt.close()

    with open(_args_draw.folder[:-1] + '_' + _args_draw.env_name, 'a') as fall:
        print(fig_name, end='|', file=fall)
        for data in regular_x_means:
            print(data, end=',', file=fall)
        print('', end='|', file=fall)
        for data in regular_y_means_mean:
            print(data, end=',', file=fall)
        print('', end='|', file=fall)
        for data in regular_y_means_var:
            print(data, end=',', file=fall)
        print('', file=fall)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw')
    parser.add_argument('--folder', default='results/', help='')
    parser.add_argument('--folder-sub', default='rewards/', help='')
    parser.add_argument('--prefix', default='', help='')
    parser.add_argument('--minmax', default='', help='')
    parser.add_argument('--max-steps', type=int, default=10e6, help='')
    parser.add_argument('--avg-num', type=int, default=10, help='')
    parser.add_argument('--team-nums', type=int, default=0, help='')
    parser.add_argument('--team-draw', type=int, default=0, help='')
    args_draw = parser.parse_args()
    names = os.listdir(args_draw.folder)
    names.sort()
    for name in names:
        name_abs = args_draw.folder + name
        if os.path.isfile(name_abs):
            continue
        args_draw.env_name = name.split('_')[0]
        print(args_draw.env_name)
        fig_curves(name_abs + '/', args_draw)
