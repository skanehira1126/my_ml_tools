import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import calc_sturges

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

def show_prediction_binary(true, pred):
    """
    plot hitgram of prediction values and true values

    Parameters
    -----
    pred : Series, 1d-array, or list.
        prediction score
    true :
        true label
    """

    plot_data = pd.DataFrame([true, pred], index=["true", "pred"]).T

    bins = calc_sturges(plot_data)
    sns.distplot(plot_data[plot_data.true == 1], label="Pos", bins=bins)
    sns.distplot(plot_data[plot_data.true == 0], label="Neg", bins=bins)
    plt.show()
    plt.close()
