import numpy as np
import pandas as pd

def check_cv_result(cv_mode, asc=True):
    """
    CrossValidatorでcvした結果の確認

    Parameters
    -----
    cv_model : Transformer
        cv訓練済みのモデル
    asc : bool default True
        metricでのsortを昇順にするか降順にするか
    """
    avgMetrics = cv_model.avgMetrics
    cv_params = cv_model.getEstimatorParamMaps()

    result_list = []
    for param_dict, avg_metric in zip(cv_params, avgMetrics):
        temp_result_map = {"metric":avg_metric}
        for param, val in param_dict.items():
            temp_result_map[param.name] = val
        result_list.append(temp_result_map)

    cv_result_df = pd.DataFrame(result_list)
    cv_result_df.sort_values("metric", ascending=asc, inplace=True)
    return cv_result_df
