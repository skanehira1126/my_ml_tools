import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def label_encode(df, columns_list, le_map={}):
    """
    label encoder

    Parameters
    -----
    df : DataFrame
        encoded data
    columns_list : str or list
        encoded label name of df
    le_map : dict,  default {}
        If this is not empty, this label encoder is used.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pd.DataFrame.")
    if isinstance(columns_list, str) :
        columns_list = [columns_list]
    elif isinstance(columns_list, list) \
        or isinstance(columns_list, ndarray):
        pass
    else:
        raise TypeError("columns_list must be str, list or 1-dimentional array.")
    
    if not isinstance(le_map, dict):
        raise TypeError("le_map must be dict.")
    else:
        for key, le in le_map.items():
            if not isinstance(le, LabelEncoder):
                raise TypeError("Value of {key} is not LabelEncoder.".format(key))
    category_df = pd.DataFrame(np.empty([df.shape[0], 0]))
    category_df.index = df.index

    for col in columns_list:
        #mapに存在している場合、mapに格納されているencoderを使用
        if le_map.get(col) is not None:
            le = le_map.get(col)
        else:
            le = LabelEncoder()
            le.fit(df[col].values)

        #変形
        category_df[f"{col}_category"] = le.transform(df[col].values)
        le_map[col] = le

    return category_df, le_map

def standard_scaler(df, columns_list, scaler_map={}):
    """
    Standerd Scaler

    Parameters
    -----
    df : DataFrame
        encoded data
    columns_list : str or list
        encoded label name of df
    scaler_map : dict,  default {}
        If this is not empty, this StandardScaler is used.
    """

    #型の確認
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pd.DataFrame.")
    if isinstance(columns_list, str) :
        columns_list = [columns_list]
    elif isinstance(columns_list, list) \
        or isinstance(columns_list, ndarray):
        pass
    else:
        raise TypeError("columns_list must be str, list or 1-dimentional array.")
    
    if not isinstance(scaler_map, dict):
        raise TypeError("scaler_map must be dict.")
    else:
        for key, scaler in scaler_map.items():
            if not isinstance(scaler, StandardScaler):
                raise TypeError("Value of {key} is not StandardScaler.".format(key))
                
    scaler_df = pd.DataFrame(np.empty([df.shape[0], 0]))
    scaler_df.index = df.index

    for col in columns_list:
        #mapに存在している場合、mapに格納されているencoderを使用
        if scaler_map.get(col) is not None:
            scaler = scaler_map.get(col)
        else:
            scaler = StandardScaler()
            scaler.fit(df[col].values.reshape(-1, 1))
            
        scaler_df[f"scaled_{col}"] = scaler.transform(df[col].values.reshape(-1, 1))
        scaler_map[col] = scaler

    return scaler_df, scaler_map
