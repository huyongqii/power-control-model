import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    """数据处理器：加载和预处理数据"""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        
    def load_and_prepare_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载并准备训练和测试数据
        
        Args:
            data_path: 处理好的数据文件路径
        Returns:
            X_train, X_test, y_train, y_test
        """
        # 加载数据
        df = pd.read_csv(data_path)
        
        # 提取特征
        features = [
            'nb_idle',             # 空闲节点数
            'nb_computing',        # 计算中的节点数
            'utilization_rate',    # 利用率
            'running_jobs',        # 运行中的作业数
            'waiting_jobs',        # 等待中的作业数
            'epower',             # 当前实际功率消耗
            'wattmin'             # 基准功率
        ]
        
        # 添加时间特征
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour / 24.0
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek / 6.0
        features.extend(['hour', 'day_of_week'])
        
        # 打印数据基本信息
        print("\nDataset Info:")
        print(f"Total samples: {len(df)}")
        print("\nFeature ranges:")
        for feature in features:
            print(f"{feature:20} min: {df[feature].min():10.2f} max: {df[feature].max():10.2f} mean: {df[feature].mean():10.2f}")
        
        # 检查是否有缺失值
        missing_values = df[features].isnull().sum()
        if missing_values.any():
            print("\nWarning: Missing values detected:")
            print(missing_values[missing_values > 0])
        
        X = df[features].values
        y = df['nb_computing'].values.reshape(-1, 1)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 特征缩放
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # 标签缩放
        y_train = self.y_scaler.fit_transform(y_train)
        y_test = self.y_scaler.transform(y_test)
        
        return X_train, X_test, y_train, y_test
    
    def inverse_transform_y(self, y_scaled):
        """将缩放后的标签转换回原始尺度"""
        return self.y_scaler.inverse_transform(y_scaled) 