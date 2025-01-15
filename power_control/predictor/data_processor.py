import pandas as pd
import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    """数据处理器：加载和预处理数据"""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.n_splits = config.get('n_splits', 5)  # 交叉验证折数
        
    def load_and_prepare_data(self, data_path: str) -> Dict[str, np.ndarray]:
        """
        加载并准备训练、验证和测试数据
        
        Args:
            data_path: 处理好的数据文件路径
        Returns:
            包含训练集、验证集和测试集的字典
        """
        # 加载数据
        df = pd.read_csv(data_path)
        
        # 提取有效特征
        features = [
            'nb_idle',             # 空闲节点数
            'utilization_rate',    # 利用率
            'running_jobs',        # 运行中的作业数
            'hour',
            'day_of_week'
        ]
        
        # 打印数据基本信息
        print("\nDataset Info:")
        print(f"Total samples: {len(df)}")
        print("\nFeature ranges:")
        for feature in features:
            print(f"{feature:20} min: {df[feature].min():10.2f} max: {df[feature].max():10.2f} mean: {df[feature].mean():10.2f}")
        
        X = df[features].values
        y = df['nb_computing'].values.reshape(-1, 1)
        
        # 首先分割出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 从剩余数据中分割出验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.2, random_state=42
        )
        
        # 特征缩放
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # 标签缩放
        y_train = self.y_scaler.fit_transform(y_train)
        y_val = self.y_scaler.transform(y_val)
        y_test = self.y_scaler.transform(y_test)
        
        # 创建交叉验证折
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        cv_splits = []
        
        for train_idx, val_idx in kf.split(X_train):
            cv_splits.append({
                'train_idx': train_idx,
                'val_idx': val_idx
            })
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'cv_splits': cv_splits,
            'feature_names': features
        }
    
    def get_cv_fold(self, fold_idx: int, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取指定交叉验证折的数据"""
        if fold_idx >= self.n_splits:
            raise ValueError(f"fold_idx must be less than {self.n_splits}")
            
        train_idx = data['cv_splits'][fold_idx]['train_idx']
        val_idx = data['cv_splits'][fold_idx]['val_idx']
        
        X_train = data['X_train'][train_idx]
        y_train = data['y_train'][train_idx]
        X_val = data['X_train'][val_idx]
        y_val = data['y_train'][val_idx]
        
        return X_train, X_val, y_train, y_val
    
    def inverse_transform_y(self, y_scaled):
        """将缩放后的标签转换回原始尺度"""
        return self.y_scaler.inverse_transform(y_scaled) 