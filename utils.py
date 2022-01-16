import math

from hyperopt import hp


def hprange(name, start, stop, step=1):
    interval = stop - start
    stop = math.ceil(interval / step)
    return start + hp.randint(name, stop + 1) * step
