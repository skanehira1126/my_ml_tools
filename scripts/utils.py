import numpy as np
from sklearn.metrics import confusion_matrix

def calc_sturges(arr):
    """
    bins計算用関数
    """
    sturges = lambda x: int(np.ceil(np.log2(x*2)))

    return sturges(len(arr))

def calc_acc(c_mat):
    return (c_mat[0, 0] + c_mat[1, 1]) / np.sum(c_mat)

def calc_acc_threthold(true_label, predict_proba, threthold=0.5):
    predict_label = np.where(predict_proba >= threthold, 1, 0)
    c_mat = confusion_matrix(true_label, predict_label)
    acc = calc_acc(c_mat)
    return acc
