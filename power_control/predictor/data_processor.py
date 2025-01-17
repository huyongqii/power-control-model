from config import MODEL_CONFIG

import torch
import holidays
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    def __init__(self, past_hour_features, cur_datetime_features, dayback_features, targets):
        self.past_hour_features = torch.FloatTensor(past_hour_features)
        self.cur_datetime_features = torch.FloatTensor(cur_datetime_features)
        self.dayback_features = torch.FloatTensor(dayback_features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'past_hour': self.past_hour_features[idx],
            'cur_datetime': self.cur_datetime_features[idx],
            'dayback': self.dayback_features[idx],
            'target': self.targets[idx]
        }

class MyDataLoader:
    def __init__(self):
        pass
    
    def create_data_loaders(self, data_dict: dict, batch_size: int) -> dict:
        """
        创建数据加载器
        
        参数:
            data_dict (dict): 包含训练、验证和测试数据的字典
            batch_size (int): 批次大小
            
        返回:
            dict: 包含数据加载器的字典
        """
        loaders = {}
        
        for split in ['train', 'val', 'test']:
            loaders[split] = self.create_one_data_loader(data_dict, batch_size, split)
        
        return loaders
    
    def create_one_data_loader(self, data_dict: dict, batch_size: int, split: str) -> dict:
        data_set = TimeSeriesDataset(
            data_dict[f'X_{split}'][0],  
            data_dict[f'X_{split}'][1],  
            data_dict[f'X_{split}'][2],  
            data_dict[f'y_{split}']
        )

        return DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

class DataProcessor:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.dayback_scaler = MinMaxScaler()
        self.cn_holidays = holidays.CN()
        
        self.pst_hour_feature_names = [
            'running_jobs',
            'waiting_jobs',
            'nb_computing',
            'utilization_rate'
        ]

        self.feature_size = len(self.pst_hour_feature_names)    

    def get_day_period(self, hour):
        """将一天分为不同时段"""
        if 5 <= hour < 9:
            return 0  # 早晨
        elif 9 <= hour < 12:
            return 1  # 上午
        elif 12 <= hour < 14:
            return 2  # 中午
        elif 14 <= hour < 18:
            return 3  # 下午
        elif 18 <= hour < 22:
            return 4  # 晚上
        else:
            return 5  # 深夜

    def is_holiday(self, date):
        """判断是否为节假日"""
        return date in self.cn_holidays

    def prepare_time_series_data(self, df):
        """
        准备时间序列数据
        
        参数:
            df (pd.DataFrame): 输入数据框
            
        返回:
            tuple: 包含处理后的历史数据、时间特征、历史模式和目标值的元组
        """
        lookback = self.config['lookback_minutes']
        forecast_horizon = self.config['forecast_minutes']
        
        past_hour_sequences = []  # 存储历史序列数据
        cur_datetime_feature_vectors = []  # 存储时间特征
        dayback_feature_vectors = []  # 存储历史模式特征
        target_min_values = []  # 存储目标值
        target_max_values = []  # 存储目标值
        
        timestamps = pd.to_datetime(df['datetime'])
        
        for i in range(len(df) - lookback - forecast_horizon + 1):
            # 1. 提取历史序列数据
            past_hour_data = df[self.pst_hour_feature_names].iloc[i:(i + lookback)].values
            
            # 2. 获取目标时间点
            target_start = i + lookback
            target_end = target_start + forecast_horizon
            target_period = df['nb_computing'].iloc[target_start:target_end].values
            
            target_min = np.min(target_period)
            target_max = np.max(target_period)
            target_time = timestamps[target_end - 1]
            
            # 3. 生成时间特征
            cur_datetime_features = self._create_time_features(target_time)
            
            # 4. 获取历史模式特征
            dayback_features = self._get_dayback_features(
                df, timestamps, target_time, i + lookback, 'nb_computing'
            )
            
            # 5. 存储数据
            past_hour_sequences.append(past_hour_data)
            cur_datetime_feature_vectors.append(cur_datetime_features)
            dayback_feature_vectors.append(dayback_features)
            target_min_values.append(target_min)
            target_max_values.append(target_max)
        
        return (np.array(past_hour_sequences),
                np.array(cur_datetime_feature_vectors),
                np.array(dayback_feature_vectors),
                np.array(target_min_values).reshape(-1, 1),
                np.array(target_max_values).reshape(-1, 1))

    def _create_time_features(self, timestamp):
        """
        创建时间特征向量
        
        参数:
            timestamp (datetime): 时间戳
            
        返回:
            list: 时间特征向量
        """
        return [
            timestamp.hour / 24.0,              # 小时 (归一化到 0-1)
            timestamp.minute / 60.0,            # 分钟 (归一化到 0-1)
            timestamp.dayofweek / 7.0,          # 星期几 (归一化到 0-1)
            float(timestamp.dayofweek >= 5),     # 是否周末
            float(self.is_holiday(timestamp.date())),  # 是否节假日
            self.get_day_period(timestamp.hour) / 6.0  # 一天中的时段 (归一化到 0-1)
        ]

    def _get_dayback_features(self, df, timestamps, target_time, current_idx, target_col):
        """
        获取历史同期时间段的特征模式，记录前后半小时的平均值
        
        参数:
            df (pd.DataFrame): 数据框
            timestamps (pd.Series): 时间戳序列
            target_time (datetime): 目标时间点
            current_idx (int): 当前索引
            target_col (str): 目标列名
            
        返回:
            np.array: 一维的历史模式特征数组，维度为4 (4天的平均值)
        """
        pattern_features = []
        window_minutes = 30  # 半小时窗口
        
        # 获取当前时间点的值作为默认值
        current_value = float(df[target_col].iloc[current_idx])
        
        # 获取不同天数的历史同期数据
        for days_back in [1, 3, 5, 7]:  # 1天、3天、5天、7天前
            minutes_back = days_back * 24 * 60
            historical_center_idx = current_idx - minutes_back
            
            if historical_center_idx >= window_minutes and historical_center_idx + window_minutes < len(df):
                # 计算前后一小时的平均值
                window_avg = df[target_col].iloc[historical_center_idx - window_minutes:historical_center_idx + window_minutes].mean()
                pattern_features.append(float(window_avg))
            else:
                # 如果历史数据不可用，使用当前值填充
                pattern_features.append(current_value)
        
        # 确保返回的是一维数组，长度为4 (4天的平均值)
        return np.array(pattern_features, dtype=np.float32)

    def load_and_prepare_data(self) -> dict:
        """
        加载并准备模型训练所需的数据
        
        参数:
            data (pd.DataFrame): 输入数据框
            
        返回:
            dict: 包含处理后的训练、验证和测试数据的字典
        """
        print("正在加载数据")

        # 1. 加载数据
        data = pd.read_csv(self.config['data_path'])

        # 确保数据按时间排序
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # 准备时间序列数据
        past_hour_features, cur_datetime_features, dayback_features, target_min_values, target_max_values = \
            self.prepare_time_series_data(data)
        
        print(f"历史序列: {past_hour_features.shape}")
        print(f"时间特征: {cur_datetime_features.shape}")
        print(f"模式特征: {dayback_features.shape}")
        print(f"目标值: {target_min_values.shape}")
        
        # 划分数据集
        train_size = int(len(target_min_values) * 0.7)
        val_size = int(len(target_min_values) * 0.15)
        
        # 训练集
        X_train = [
            past_hour_features[:train_size],
            cur_datetime_features[:train_size],
            dayback_features[:train_size]
        ]
        y_train = np.hstack([target_min_values[:train_size], target_max_values[:train_size]])
        
        # 验证集
        X_val = [
            past_hour_features[train_size:train_size + val_size],
            cur_datetime_features[train_size:train_size + val_size],
            dayback_features[train_size:train_size + val_size]
        ]
        y_val = np.hstack([target_min_values[train_size:train_size + val_size],
                           target_max_values[train_size:train_size + val_size]])
        
        # 测试集
        X_test = [
            past_hour_features[train_size + val_size:],
            cur_datetime_features[train_size + val_size:],
            dayback_features[train_size + val_size:]
        ]
        y_test = np.hstack([target_min_values[train_size + val_size:],
                           target_max_values[train_size + val_size:]])

        # 对不同类型的特征进行缩放
        # 历史特征
        X_train[0] = self.feature_scaler.fit_transform(
            X_train[0].reshape(-1, self.feature_size)
        ).reshape(X_train[0].shape)
        X_val[0] = self.feature_scaler.transform(
            X_val[0].reshape(-1, self.feature_size)
        ).reshape(X_val[0].shape)
        X_test[0] = self.feature_scaler.transform(
            X_test[0].reshape(-1, self.feature_size)
        ).reshape(X_test[0].shape)
        
        # 历史模式特征
        X_train[2] = self.dayback_scaler.fit_transform(X_train[2])
        X_val[2] = self.dayback_scaler.transform(X_val[2])
        X_test[2] = self.dayback_scaler.transform(X_test[2])
        
        # 目标值缩放
        y_train_min = self.target_scaler.fit_transform(y_train[:, 0].reshape(-1, 1))
        y_train_max = self.target_scaler.transform(y_train[:, 1].reshape(-1, 1))
        y_val_min = self.target_scaler.transform(y_val[:, 0].reshape(-1, 1))
        y_val_max = self.target_scaler.transform(y_val[:, 1].reshape(-1, 1))
        y_test_min = self.target_scaler.transform(y_test[:, 0].reshape(-1, 1))
        y_test_max = self.target_scaler.transform(y_test[:, 1].reshape(-1, 1))

        y_train = np.hstack([y_train_min, y_train_max])
        y_val = np.hstack([y_val_min, y_val_max])
        y_test = np.hstack([y_test_min, y_test_max])

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

    def inverse_transform_y(self, y_scaled):
        """将缩放后的标签转换回原始尺度"""
        min_values = self.target_scaler.inverse_transform(y_scaled[:, 0].reshape(-1, 1))
        max_values = self.target_scaler.inverse_transform(y_scaled[:, 1].reshape(-1, 1))
        return np.hstack([min_values, max_values])
