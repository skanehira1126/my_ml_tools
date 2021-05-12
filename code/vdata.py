import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import calc_sturges

from IPython.display import display

def show_corr(df, corr_name='pearson', show=True):
    """
    相関係数の計算と描画

    Parameters
    -----
    df : pandas.DataFrame
        相関係数を計算する対象のデータフレーム
    corr_name : str, default pearson
        相関係数の種類
    show : bool, default True
        ヒートマップを描画するかどうか
    """

    #相関係数を計算
    df_corr = df.corr(corr_name)
    display(df_corr)

    #ヒートマップで描画
    if show:
        sns.heatmap(df_corr, vmax=1, vmin=-1, center=0)
        plt.show()
        plt.close()


