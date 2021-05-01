from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from .model import MetaModel
from .utils import Util

class ModelLgb(MetaModel):
    
    """
    LightGBMのラッパークラス

    Attributes
    -----
    name : str
        モデルの識別子
    params : dict
        モデルのハイパーパラメータ
    metric_log : dict
        学習過程を保存する辞書
    _num_iteration : int
        モデルのiteration数
    """

    def __init__(self, name, params):
        super().__init__(f"lgb-{name}", params)
        
        self.metric_log = {}
        self._num_iteration = -1
    
    def train(self, tr_x, tr_y, va_x=None, va_y=None):
        
        if va_x is None and va_y is None: #validationなし
            validate = False
        elif va_x is not None and va_y is not None: #validationあり
            validate = True
        else: #エラー
            raise ValueError("Both va_x and va_y must be None or not None.")
        
        #データの作成
        lgb_tr = lgb.Dataset(tr_x, label=tr_y)
        if validate:
            lgb_va = lgb.Dataset(va_x, label=va_y)
        
        #ハイパーパラメータ
        params = dict(self.params)
        num_iterations = params.pop("num_iterations")
        try:
            verbose_eval = params.pop("verbose_eval") 
        except :
            verbose_eval = 10

        #モデルの訓練
        if validate:
            valid_sets=[lgb_tr, lgb_va]
            valid_names=['train', 'valid']
            early_stopping_rounds=params.pop("early_stopping_rounds")
            
            self.model = lgb.train(
                params
                , train_set=lgb_tr
                , valid_sets=valid_sets
                , valid_names=valid_names
                , early_stopping_rounds=early_stopping_rounds
                , verbose_eval=verbose_eval
                , evals_result=self.metric_log
            )
            
            self._num_iteration = self.model.best_iteration
        else:
            #validationデータの追加
            valid_sets=[lgb_tr]
            valid_names=['train']
            
            self.model = lgb.train(
                params
                , train_set=lgb_tr
                , valid_set=valid_set
                , valid_names=valid_names
                , verbose_eval=verbose_eval
                , evals_result=self.metric_log
            )
            self._num_iteration = self.model.current_iteration()
        
    def predict(self, test_x: pd.DataFrame):
        predict_val = self.model.predict(test_x, num_iteration=self._num_iteration)
        
        return predict_val
    
    def save_model(self):
        model_path = Path(f"./model/{self.name}.model")
        #pickleで保存
        Util.dump(self.model, model_path)
    
    def load_model(self, path):
        model_path = Path(f"./model/{self.name}.model")
        self.model = Util.load(path)
        
        self._num_iteration = self.model.best_iteration if self.best_iteration <= 0 else self.current_iteration()
   
    def feature_importance(self, importance_type:str="gain") -> np.ndarray:
        """
        モデルのfeature importanceを返却する

        Parameters
        -----
        importance_type : str, optional(default="gain")
            importanceの種類
        
        Returns
        -----
        importance : numpy.ndarray
            importanceの値
        """

        return self.model.feature_importance(importance_type=importance_type, iteration=self._num_iteration)

