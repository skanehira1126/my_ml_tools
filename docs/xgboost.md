#XGBoost
## Docs

* [xgboost](https://xgboost.readthedocs.io/en/latest/index.html)
* Parameters : [https://xgboost.readthedocs.io/en/latest/parameter.html](https://xgboost.readthedocs.io/en/latest/parameter.html)
    * [公式ドキュメント](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)
    * [Analyics Vidhya](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)：英語だけど結構詳しく書いてある。
    * [パラメータについて](https://sites.google.com/view/lauraepp/parameters)：パラメータについて

    
## パラメータ

### チューニング
主にチューニングの対象となるパラメータ

* `eta` : 学習率
* `max_depth` : 木の深さ
* `num_round` : 作成する決定木の本数
* `min_child_weight` : 葉を分岐するために葉を構成する最小限の2階微分値（≒データ数）
* `gamma` : 決定木を分岐するために最低限減らさなくてはならない目的変数の値
* `colsample_bytree` : 決定木ごとに特徴量の列をサンプリングする割合
* `subsample` : 決定木ごとに学習データの行をサンプリングする割合
* `alpha` : 葉のウェイトに対するL1正則化の係数
* `lambda` : 葉のウェイトに対するL2正則化の係数

#### 主な手順
1. 学習を制御するパラメータの調整
    * `eta` : 小さくしすぎると学習が収束するまで時間がかかる
    * `num_round` : 最初は大きめに設定してアーリーストッピングでいい
2. 重要なパラメータを最適化
    * `max_depth` : 深くするとモデルの表現度が上がる
    * `subsample` : ランダム性による過学習の抑制（データ方向）
    * `colsample_bytree` : ランダム性による過学習の抑制（特徴量方向）
    * `min_child_weight` : 分岐のしやすさを調整し、モデルの表現度を抑える
3. その他のパラメータで微調整
    * `gamma` : 分岐のしやすさを調整し、モデルの表現度を抑える
    * `alpha`, `lambda` : 正則化による過学習の抑制