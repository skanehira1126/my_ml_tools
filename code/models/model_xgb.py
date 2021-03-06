from __future__ import annotations
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

from .model import MetaModel
from .utils import Util

class ModelXgb(MetaModel):
    
    """
    XGBoostのラッパークラス

    Attributes
    -----
    name : str
        モデルの識別子
    params : dict
        モデルのハイパーパラメータ
    metric_log : dict
        学習過程を保存する辞書
    """

    def __init__(self, name:str, params:dict) -> None:
        """
        初期化処理
        
        Paramters
        -----
        name: str
            モデル名
        params: dict
            モデルの学習に用いるパラメーター
        """
        super().__init__(f"xgb-{name}", params)
        
        self.metric_log = {}
        self.model = None
    
    def train(self, tr_x:pd.DataFrame, tr_y:pd.DataFrame, va_x:pd.DataFrame=None, va_y:pd.DataFrame=None) -> None:
        """
        モデルの学習を行う。
        
        Parameters
        -----
        tr_x, tr_y : pd.DataFrame
            学習用データの特徴量とラベル
        va_x, va_y : pd.DataFrame
            検証用データの特徴量とラベル
            
        Note
        -----
        paramsからモデルのパラメータとしてpopされる
        * num_boost_round 
        * verbose_eval 
        * early_stopping_rounds
        * categorical_feature 
        """
        #初期化
        self.metric_log = {}
        
        if va_x is None and va_y is None: #validationなし
            validate = False
        elif va_x is not None and va_y is not None: #validationあり
            validate = True
        else: #エラー
            raise ValueError("Both va_x and va_y must be None or not None.")
        
        #データの作成
        dtrain = xgb.DMatrix(tr_x, label=tr_y)
        if validate:
            dvalid = xgb.DMatrix(va_x, label=va_y)
        
        #=====ハイパーパラメータ
        params = dict(self.params)
        #train関数の引数
        num_boost_round = params.pop("num_boost_round")
        verbose_eval = params.pop("verbose_eval") if params.get("verbose_eval") is not None else 10
        early_stopping_rounds= params.pop("early_stopping_rounds") if params.get("early_stopping_rounds") is not None else None

        #モデルの訓練
        if validate:
            evals = [(dtrain, "train"), (dvalid, "valid")]
            
            self.model = xgb.train(
                params
                , dtrain
                , num_boost_round=num_boost_round
                , evals=evals
                , early_stopping_rounds=early_stopping_rounds
                , verbose_eval = verbose_eval
                , evals_result=self.metric_log
            )
            
        else:
            evals = [(dtrain, "train")]
            
            self.model = xgb.train(
                params
                , dtrain
                , num_boost_round=num_boost_round
                , evals=evals
                , verbose_eval = verbose_eval
                , evals_result=self.metric_log
            )
        
    def predict(self, test_x: pd.DataFrame) -> np.ndarray:
        """
        予測を行う
        
        Parameters
        -----
        test_x : pd.DataFrame
            予測する特徴量データ
            
        Returns
        predict_val : np.ndarray
            モデルによる予測値
        """
        dtest = xgb.DMatrix(test_x)
        predict_val = self.model.predict(dtest, ntree_limit=self.model.best_iteration)
        
        return predict_val
    
    def save_model(self) -> None:
        """
        モデルを保存する
        
        Note
        -----
        best_iterationを残すためにpkl形式で保存
        """
        model_path = pathlib.Path(f"../model/{self.name}.model")
        #pickleで保存
        Util.dump(self, model_path)
    
    @classmethod
    def load_model(cls, path:pathlib.Path) -> __class__:
        """
        モデルを読み込む
        
        Parameters
        -----
        path : pathlib.Path or str
            path of model file
            
        Returns
        -----
        model : ModelXgb
            指定されたモデルファイルを読み込んだもの
        """
        cls = Util.load(path)
        return cls
   
    def feature_importance(self, importance_type:str="total_gain") -> pd.DataFrame:
        """
        モデルのfeature importanceを返却する

        Parameters
        -----
        importance_type : str, optional(default="gain")
            importanceの種類
        
        Returns
        -----
        df_importance :pd.DataFrame
            importanceの値
        """
        f_importance_map = defaultdict(int)
        f_importance_map.update(self.model.get_score(importance_type=importance_type))

        #DataFrameに変換
        df_importance = pd.DataFrame(
            [[col, f_importance_map[col]] for col in self.model.feature_names]
            , columns=["feature_name", importance_type]
        )
        return df_importance
