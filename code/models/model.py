import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

class MetaModel(metaclass=ABCMeta):
    """
    各モデルの基底となるクラス
    
    Attributes
    -----
    name : str
        モデルを管理する識別子
    params : dict
        ハイパーパラメータ
    model : 
        ラップしたモデル本体
    """
    
    def __init__(self, name: str,  params: dict) -> None:
        """
        コンストラクタ
        
        Parameters
        -----
        name : str
            モデルを管理する識別子
        params : dict
            ハイパーパラメータ
        """
        
        self.name = name
        self.params = params
        self.model = None
    
    @abstractmethod
    def train(self, tr_x: pd.DataFrame, tr_y: pd.DataFrame
              , va_x: pd.DataFrame=None, va_y: pd.DataFrame=None) :
        """
        モデルの学習を行う関数
        
        Parameters 
        -----
        tr_x : pd.DataFrame
            学習データの特徴量
        tr_y : pd.DataFrame
            学習データの目的変数
        va_x : pd.DataFrame
            検証データの特徴量
        va_y : pd.DataFrame
            検証データの目的変数
        """
        pass

    @abstractmethod
    def predict(self, test_x: pd.DataFrame) -> np.ndarray:
        """
        学習済みのモデルで与えられたデータに対して予測値を返す
        
        Parameters
        -----
        test_x : pd.DataFrame
            モデルが予測を行う特徴量
        
        Returns
        -----
        predict_val : np.ndarray
            モデルの予測値
        """
        pass
    
    @abstractmethod
    def save_model(self):
        """
        modelの保存を行う
        """
        
    @abstractmethod
    def load_model(self):
        """
        modelの読み込みを行う
        """
    