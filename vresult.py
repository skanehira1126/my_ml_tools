import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve

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

def show_roc_curve(true_value, predict_value, pos_label=1):
    """
    plot roc curve

    Parameters
    -----
    true_value : array-like (1d)
        true value
    predict_value : list or array-like (1d or 2d) 
        predicted value
    true_label : int
        label of true
    """
    
    if isinstance(predict_value, list):
        pred = np.array(predict_value)
    elif isinstance(predict_value, np.ndarray):
        pred = predict_value
    else:
        raise ValueError("predict value ")
    #When predict value is 1-dimentional array, convert to 2-dimentional array
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
            
    if len(true_value) != pred.shape[1]:
        raise ValueError("The number of true value and predict value must be the same.")
    
    #visualize
    plt.figure(figsize=(12, 8))
    for p in pred:
        #roc curve の計算
        fpr, tpr, _ = roc_curve(true_value, p, pos_label=pos_label)
        plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive rate", size=20)
    plt.ylabel("True Positive rate", size=20)
    plt.show()
    plt.close()
