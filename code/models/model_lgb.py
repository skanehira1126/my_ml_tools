from __future__ import annotations
import pathlib

import numpy as np
import pandas as pd
import lightgbm as lgb

from model import MetaModel
from utils import Util

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
    """

    def __init__(self, name:str, params:dict):
        """
        初期化処理
        
        Paramters
        -----
        name: str
            モデル名
        params: dict
            モデルの学習に用いるパラメーター
        """
        super().__init__(f"lgb-{name}", params)
        
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
        lgb_tr = lgb.Dataset(tr_x, label=tr_y)
        if validate:
            lgb_va = lgb.Dataset(va_x, label=va_y)
        
        #=====ハイパーパラメータ
        params = dict(self.params)
        #train関数の引数
        num_boost_round = params.pop("num_boost_round")
        verbose_eval = params.pop("verbose_eval") if params.get("verbose_eval") is not None else 10
        early_stopping_rounds= params.pop("early_stopping_rounds") if params.get("early_stopping_rounds") is not None else None
        categorical_feature = params.pop("categorical_feature") if params.get("categorical_feature") is not None else "auto"

        #モデルの訓練
        if validate:
            valid_sets=[lgb_tr, lgb_va]
            valid_names=['train', 'valid']
            
            self.model = lgb.train(
                params
                , train_set=lgb_tr
                , num_boost_round=num_boost_round
                , valid_sets=valid_sets
                , valid_names=valid_names
                , early_stopping_rounds=early_stopping_rounds
                , verbose_eval=verbose_eval
                , evals_result=self.metric_log
            )
            
        else:
            #validationデータの追加
            valid_sets=[lgb_tr]
            valid_names=['train']
            
            self.model = lgb.train(
                params
                , train_set=lgb_tr
                , num_boost_round=num_boost_round
                , valid_sets=valid_sets
                , valid_names=valid_names
                , verbose_eval=verbose_eval
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
        predict_val = self.model.predict(test_x, num_iteration=self.model.best_iteration)
        
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
        model : ModelLgb
            指定されたモデルファイルを読み込んだもの
        """
        cls = Util.load(path)
        return cls
   
    def feature_importance(self, importance_type:str="gain") -> pd.DataFrame:
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
        feature_name = self.model.feature_name()
        feature_importance = self.model.feature_importance(importance_type=importance_type, iteration=self.model.best_iteration)

        #DataFrameに変換
        df_importance = pd.DataFrame(
            [feature_name, feature_importance]
            , index=["feature_name", importance_type]
        ).T
        
        return df_importance