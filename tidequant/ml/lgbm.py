import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

import tidemodel as tm
from .backtest import backtest
from .data import StockDataSource
from .utils import get_logger, reset_logger, set_seed


class LgbmExperiment(Experiment):
    
    params: Dict[str, Any] = {
        "num_threads": -1,
        "seed": 42,
        "deterministic": True,

        "objective": "regression",
        "metric": "None",

        "boosting": "gbdt",
        "num_iterations": 300,
        "learning_rate": 0.13,

        # "max_bin": 31,
        # "min_data_in_bin": 1000,

        "max_depth": -1,
        "num_leaves": 144,
        # "min_data_in_leaf": 10000,

        "bagging_freq": 1,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
    }

    def __init__(
        self,
        folder: str,
        create_folder: bool = True,
        params: Dict[str, Any] = {},
    ) -> None:
        super().__init__(
            folder=folder,
            create_folder=create_folder,
            params=params,
        )

        # 设置lgb的logger
        lgb.register_logger(self.logger)

        # 数据
        self.x_names: List[str] = None
        self.y_name: str = None

        self.train_dataset: lgb.Dataset = None
        self.train_y: pd.DataFrame = None

        self.test_dataset: lgb.Dataset = None
        self.test_x: pd.DataFrame = None
        self.raw_test_y: pd.DataFrame = None
        self.test_y: pd.DataFrame = None

        # 模型
        self.model: lgb.Booster = None

    def load_data(
        self,
        df_folder: str,
        x_names: List[str],
        y_name: str,
        train_start_dt: str,
        train_end_dt: str,
        test_start_dt: str,
        test_end_dt: str,
        frac: float = 1.0,
        clip_pct: float = 0.5,
    ) -> None:
        """
        基于DataFrame数据库加载数据集
        """
        self.logger.info(f"load data from {df_folder}")

        data_source = StockDataSource(df_folder)
        self.x_names = x_names
        self.y_name = y_name

        train_dts: List[str] = data_source.load_dates(
            train_start_dt, train_end_dt
        )
        train_x, self.train_y = data_source.load_train_data(
            train_dts,
            factor_names=x_names,
            label_names=y_name,
            frac=frac,
            seed=self.params["seed"],
        )

        test_dts: List[str] = data_source.load_dates(test_start_dt, test_end_dt)
        self.test_x, self.raw_test_y = data_source.load_train_data(
            test_dts,
            factor_names=x_names,
            label_names=y_name,
            frac=frac,
            seed=self.params["seed"],
        )

        # 在数据集没有构建前可以修改原始数据  
        if clip_pct > 0.0:
            lo, hi = np.nanpercentile(self.train_y, [clip_pct, 100 - clip_pct])
            self.train_y = np.clip(self.train_y, lo, hi)
            self.test_y = np.clip(self.raw_test_y, lo, hi)
        else:
            self.test_y = self.raw_test_y

        self.train_dataset = lgb.Dataset(
            train_x,
            label=self.train_y,
            feature_name=self.x_names,
        )
        self.test_dataset = lgb.Dataset(
            self.test_x,
            label=self.test_y,
            feature_name=self.x_names,
        )
        self.logger.info(f"{len(self.train_y)} train, {len(self.test_y)} test")

    def train(self, ) -> None:
        """
        训练模型
        """
        self.model = lgb.train(
            self.params,
            self.train_dataset,
            valid_sets=self.test_dataset,
            feval=self._evaluate_ic,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=1),
            ],
        )

        # 储存模型的预测结果
        test_y_pred: np.ndarray = self.model.predict(
            self.test_x,
            num_iteration=self.model.best_iteration,
        )
        test_y: pd.DataFrame = self.test_y.to_frame("y")
        test_y["y_pred"] = test_y_pred
        test_y["y_raw"] = self.raw_test_y
        test_y.to_parquet(
            os.path.join(self.folder, "test_y.parquet"), index=True
        )

        # 储存成回测支持的格式
        self.model.save_model(
            os.path.join(self.folder, "model_cluster_0.json"),
            num_iteration=self.model.best_iteration
        )
        
        with open(
            os.path.join(self.folder, "cluster"), "w", encoding="utf-8"
        ) as f:
            pass

        tm.utils.write_txt(
            os.path.join(self.folder, "factornames_0"), self.x_names
        )

        with open(
            os.path.join(self.folder, "parameters_0"), "w", encoding="utf-8"
        ) as f:
            f.write("0.2\n0.09\n0.045\n1.0")

    def _evaluate_ic(self, y_pred: np.ndarray, dataset: lgb.Dataset) -> Tuple:
        return "ic", tm.data.np_ic(self.test_y, y_pred), True

    def test(self, *args, **kwargs) -> Dict[float, Dict[str, float]]:
        """
        在测试集上运行回测
        """
        results = backtest(self.folder, *args, **kwargs)
        with open(
            os.path.join(self.folder, "results.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=2)

        # self.logger.info(f"results are {results}")
        return results

    def compute_importance(self, topk: int) -> pd.DataFrame:
        """
        根据训练结果计算重要度得分
        """
        feature_names = self.model.feature_name()

        # 计算gain得分
        gain_importance = self.model.feature_importance(importance_type="gain")

        # 计算shap得分
        # 采样100万样本用于计算
        rng = np.random.default_rng(self.params["seed"])
        n: int = len(self.test_x)
        idx: np.ndarray = rng.choice(n, min(1000000, n), replace=False)

        contrib = self.model.predict(
            self.test_x[idx],
            num_iteration=self.model.best_iteration,
            pred_contrib=True
        )
        shap_importance = np.mean(np.abs(contrib[:, : -1]), axis=0)

        importance_df = pd.DataFrame({
            "gain": gain_importance,
            "shap": shap_importance,
        }, index=feature_names)
        importance_df.to_csv(
            os.path.join(self.folder, "importance_df.csv")
        )

        # 保存前topk个因子
        importance_df = importance_df.sort_values(by="shap", ascending=False)
        x = importance_df.index[: topk].tolist()
        tm.utils.write_txt(os.path.join(self.folder, "x_list.txt"), x)
        return importance_df
