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
