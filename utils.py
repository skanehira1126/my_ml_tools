import numpy as np

def calc_sturges(arr):
    """
    bins計算用関数
    """
    sturges = lambda x: int(np.ceil(np.log2(x*2)))

    return sturges(len(arr))
