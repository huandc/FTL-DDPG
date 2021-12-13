import time
from functools import wraps



def currentTime(str, flag=False):
    if flag:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  " + str, end="")
    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "  " + str)
    return 1


def end(str):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " + str)
    return -1


# 定义装饰器
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()

        print('step processing time [finished {func_name} in {time:.2f}s]'.format(func_name=function.__name__,
                                                                                  time=t1 - t0))
        return result

    return function_timer
