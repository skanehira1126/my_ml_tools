import pathlib
import joblib

class Util(object):
    
    """
    モデルの保存と読み込みを行うクラス
    """

    @classmethod
    def dump(cls, target_object:object, path:pathlib.Path) -> None:
        """
        モデルをpickle形式で保存する

        Parameters
        -----
        target_object : 
            保存する変数
        path : pathlib.Path
            保存するPath
        """

        path.parent.mkdir(exist_ok=True)
        joblib.dump(target_object, path, compress=True)

    @classmethod
    def load(cls, path:pathlib.Path):
        """
        モデルを読み込む

        Parameters
        -----
        path : pathlib.Path
            読み込むモデルのPath

        Returns
        -----
        model : 
            モデル
        """
        
        if path.exists():
            model = joblib.load(path)
        else:
            raise FileNotFoundError(f"{str(path)} does not exists.")
        return joblib.load(path)
