import functools
import time


def functimer(func):
    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        print(func.__name__, "start...")
        tic = time.time()
        result = func(*args, **kwargs)
        print(func.__name__, "done. Used", time.time()-tic, "s")
        return result
    return wrapped_func

    # with open(args.exp_dir[:-1] + '_configs', 'a') as fconfigs:
    #     pprint.pprint(args, fconfigs)
    #     print(time.ctime(starttime), ' ------ ', time.ctime(time.time()), '        ',
    #           (time.time() - starttime) // 3600, 'hours', np.round((time.time() - starttime) % 3600 / 60, 1), 'minutes',
    #           file=fconfigs)
