# Hypteropt
## docs
* [github](https://github.com/hyperopt/hyperopt)

## 使い方

```python
from hyperopt import hp, tpe, Trials, fmin
```

* hq : 探索するパラメータの分布を決定するためのもの
* tpe : 探索方法の指定に利用（random or tpeロジック)
* Trials : 探索過程の記録用
* fmin : 探索を行うためのもの

```python
# 探索回数
max_evals = 200
# 探索過程
trials = Trials()

best = fmin(
    objective # 目的関数
    , param_space # 探索するパラメータのdictもしくはlist
    , algo=tpe.suggest # 探索ロジック
    , max_evals=max_evals
    , trials=trials 
    , verbose=1
)
```

### 探索空間の設定
