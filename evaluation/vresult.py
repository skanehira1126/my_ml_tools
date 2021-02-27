import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

def show_importance(f_importance, feats, normalize=True):
    """
    visualize importance

    Parameters
    -----
    f_importance : list or array
        array of importance
    feats : list
        feature names
    is_normalize : bool
        Whether to normalize the importance
    """
    if normalize:
        f_importance = f_importance / np.sum(f_importance)
    df_importance = pd.DataFrame({'feature':feats, 'importance':f_importance})
    df_importance = df_importance.sort_values('importance', ascending=False)
    display(df_importance)

    sns.barplot(x="importance", y="feature", data=df_importance)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()
    plt.close()

    return df_importance

def show_prediction_binary(pred, true):
    """
    plot hitgram of prediction values and true values

    Parameters
    -----
    pred : Series, 1d-array, or list.
        prediction score
    true :
        true label
    """
    def calc_sturges(df, col):
        """
        bins計算用関数
        """
        sturges = lambda n:  np.ceil(np.log2(n*2))

        return sturges(len(df[col]))

    plot_data = pd.DataFrame([pred, true], index=["pred", "true"]).T

    bins = calc_sturges(plot_data, "pred")
    sns.distplot(plot_data[plot_data.true == 1], label="Pos", norm_hist=norm_hist)
    sns.distplot(plot_data[plot_data.true == 0], label="Neg", norm_hist=norm_hist)
    plt.show()
    plt.close()
