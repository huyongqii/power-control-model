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

        is_shuffle = False
        if split == 'train':
            is_shuffle = True

        return DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=is_shuffle,
            num_workers=4,
            pin_memory=True
        )

class DataProcessor:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.dayback_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.cn_holidays = holidays.CN()
        
        self.pst_hour_feature_names = [
            'running_jobs',
            'waiting_jobs',
            'nb_computing',
            'utilization_rate'
            # 'epower'  # 直接使用原始功率值
        ]

        self.feature_size = len(self.pst_hour_feature_names)    
        self.base_power = 125000  # 保留基准功率值用于异常值检测

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
        elif 18 <= hour < 24:
            return 4  # 晚上
        else:
            return 5  # 深夜

    def is_holiday(self, date):
        """判断是否为节假日"""
        return date in self.cn_holidays

    def prepare_time_series_data(self, df):
        """准备时间序列数据"""
        # 1. 添加数据验证
        self._validate_input_data(df)
        
        # 2. 处理异常值
        df = self._handle_outliers(df)
        
        lookback = self.config['lookback_minutes']
        forecast_horizon = self.config['forecast_minutes']
        
        past_hour_sequences = []  # 存储历史序列数据
        cur_datetime_feature_vectors = []  # 存储时间特征
        dayback_feature_vectors = []  # 存储历史模式特征
        target_values = []  # 存储目标值（直接使用节点数量）
        
        timestamps = pd.to_datetime(df['datetime'])
        
        for i in range(len(df) - lookback - forecast_horizon + 1):
            # 1. 提取历史序列数据
            past_hour_data = df[self.pst_hour_feature_names].iloc[i:(i + lookback)].values
            
            # 2. 获取目标时间点
            target_start = i + lookback
            target_end = target_start + forecast_horizon
            target_value = df['nb_computing'].iloc[target_start]  # 直接使用目标时刻的节点数量
            
            target_time = timestamps[target_start]
            
            # 3. 生成时间特征
            cur_datetime_features = self._create_time_features(
                timestamps[target_start], 
                timestamps[target_end - 1]
            )
            
            # 4. 获取历史模式特征
            dayback_features = self._get_dayback_features(
                df, timestamps, target_time, target_start, 'nb_computing'
            )
            
            # 5. 存储数据
            past_hour_sequences.append(past_hour_data)
            cur_datetime_feature_vectors.append(cur_datetime_features)
            dayback_feature_vectors.append(dayback_features)
            target_values.append(target_value)
        
        return (np.array(past_hour_sequences),
                np.array(cur_datetime_feature_vectors),
                np.array(dayback_feature_vectors),
                np.array(target_values).reshape(-1, 1))  # 确保目标值是二维数组

    def _create_time_features(self, start_time, end_time):
        """创建时间范围的特征向量"""
        is_weekend = float(start_time.dayofweek >= 5)
        is_holiday = float(self.is_holiday(start_time.date()))
        
        hours = pd.date_range(start_time, end_time, freq='1min').hour
        period_counts = np.zeros(6)
        for hour in hours:
            period = self.get_day_period(hour)
            period_counts[period] += 1
        
        main_period = np.argmax(period_counts)
        
        return [
            is_weekend,         # 是否周末 (0/1)
            is_holiday,         # 是否节假日 (0/1)
            main_period / 6.0,  # 主要时间段 (归一化到 0-1)
        ]

    def _get_dayback_features(self, df, timestamps, target_time, current_idx, target_col):
        """获取历史模式特征的增强版本"""
        pattern_features = []
        window_minutes = 30
        
        for days_back in [1, 3, 5, 7]:
            minutes_back = days_back * 24 * 60
            historical_center_idx = current_idx - minutes_back
            
            if historical_center_idx >= window_minutes and historical_center_idx + window_minutes < len(df):
                # 获取历史时间段的数据
                historical_window = df[target_col].iloc[
                    historical_center_idx - window_minutes:
                    historical_center_idx + window_minutes
                ]
                
                # 增加更多统计特征
                pattern_features.extend([
                    float(historical_window.min()),     # 最小值
                    float(historical_window.max()),     # 最大值
                ])
            else:
                # 使用当前值填充
                current_value = float(df[target_col].iloc[current_idx])
                pattern_features.extend([current_value] * 2)  # 9个特征
        
        return np.array(pattern_features, dtype=np.float32)

    def load_and_prepare_data(self) -> dict:
        """加载并准备模型训练所需的数据"""
        print("正在加载数据")
        data = pd.read_csv(self.config['data_path'])
        data = data.sort_values('datetime').reset_index(drop=True)
        
        past_hour_features, cur_datetime_features, dayback_features, target_values = \
            self.prepare_time_series_data(data)
        
        # 确保目标值非负
        target_values = np.maximum(target_values, 0)
        
        # 划分数据集
        train_size = int(len(target_values) * 0.7)
        val_size = int(len(target_values) * 0.15)
        
        X_train = [
            past_hour_features[:train_size],
            cur_datetime_features[:train_size],
            dayback_features[:train_size]
        ]
        y_train = target_values[:train_size]
        
        X_val = [
            past_hour_features[train_size:train_size + val_size],
            cur_datetime_features[train_size:train_size + val_size],
            dayback_features[train_size:train_size + val_size]
        ]
        y_val = target_values[train_size:train_size + val_size]
        
        X_test = [
            past_hour_features[train_size + val_size:],
            cur_datetime_features[train_size + val_size:],
            dayback_features[train_size + val_size:]
        ]
        y_test = target_values[train_size + val_size:]

        # 特征缩放
        X_train[0] = self.feature_scaler.fit_transform(
            X_train[0].reshape(-1, self.feature_size)
        ).reshape(X_train[0].shape)
        X_val[0] = self.feature_scaler.transform(
            X_val[0].reshape(-1, self.feature_size)
        ).reshape(X_val[0].shape)
        X_test[0] = self.feature_scaler.transform(
            X_test[0].reshape(-1, self.feature_size)
        ).reshape(X_test[0].shape)
        
        # 历史模式特征缩放
        X_train[2] = self.dayback_scaler.fit_transform(X_train[2])
        X_val[2] = self.dayback_scaler.transform(X_val[2])
        X_test[2] = self.dayback_scaler.transform(X_test[2])
        
        # 目标值缩放
        y_train = self.target_scaler.fit_transform(y_train)
        y_val = self.target_scaler.transform(y_val)
        y_test = self.target_scaler.transform(y_test)

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

    def inverse_transform_y(self, y_scaled):
        """确保正确的反标准化"""
        original_range = self.target_scaler.data_range_
        original_min = self.target_scaler.data_min_
        
        y_original = self.target_scaler.inverse_transform(y_scaled)
        
        print(f"原始数据范围: {original_min} - {original_min + original_range}")
        print(f"反标准化后范围: {y_original.min()} - {y_original.max()}")
        
        return y_original
    
    def _validate_input_data(self, df):
        """验证输入数据的完整性和有效性"""
        # 检查必要的列是否存在
        required_columns = self.pst_hour_feature_names[:-1] + ['datetime', 'epower']  # 不检查normalized列
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        # 检查时间戳的连续性
        timestamps = pd.to_datetime(df['datetime'])
        time_diff = timestamps.diff().dropna()
        if not (time_diff == pd.Timedelta(minutes=1)).all():
            print("警告: 时间序列不连续，可能影响预测效果")
        
        # 检查功率值的合理性
        if (df['epower'] < self.base_power * 0.9).any():
            print("警告: 存在异常低的功率值")
        if (df['epower'] > self.base_power * 2).any():
            print("警告: 存在异常高的功率值")
        
        # 检查数值的有效性
        for col in self.pst_hour_feature_names:
            if df[col].isnull().any():
                print(f"警告: 列 {col} 存在空值")
            if (df[col] < 0).any():
                print(f"警告: 列 {col} 存在负值")

    def _handle_outliers(self, df):
        """处理异常值"""
        df_clean = df.copy()
        
        # 处理功率异常值
        power_mask = (df['epower'] < self.base_power * 0.9) | (df['epower'] > self.base_power * 2)
        if power_mask.any():
            print(f"发现 {power_mask.sum()} 个功率异常值")
            # 使用移动中位数填充异常值
            window_size = 5
            moving_median = df['epower'].rolling(
                window=window_size,
                center=True,
                min_periods=1
            ).median()
            df_clean.loc[power_mask, 'epower'] = moving_median[power_mask]
        
        for col in self.pst_hour_feature_names:
            if col in ['running_jobs', 'waiting_jobs', 'nb_computing']:
                # 对于作业数量，只检查负值
                outliers = df[col] < 0
                if outliers.any():
                    print(f"列 {col} 发现 {outliers.sum()} 个负值")
                    # 将负值设为0
                    df_clean.loc[outliers, col] = 0
                    
            elif col == 'utilization_rate':
                # 对于利用率，检查是否在合理范围内
                outliers = (df[col] < 0) | (df[col] > 100)
                if outliers.any():
                    print(f"列 {col} 发现 {outliers.sum()} 个异常值")
                    # 使用移动平均替换异常值
                    window_size = 5
                    moving_avg = df[col].rolling(
                        window=window_size,
                        center=True,
                        min_periods=1
                    ).mean()
                    df_clean.loc[outliers, col] = moving_avg[outliers]
            
            print(f"列 {col} 的范围: [{df_clean[col].min()}, {df_clean[col].max()}]")
        
        return df_clean
    
    def process_single_record(self, data_path):
        """
        处理单条记录的数据
        
        参数:
            data_path (str): CSV数据文件路径
            
        返回:
            tuple: (past_hour_data, cur_datetime, dayback_data)
                - past_hour_data: 形状为 [240, 5] 的 numpy 数组
                - cur_datetime: 形状为 [3] 的 numpy 数组
                - dayback_data: 形状为 [8] 的 numpy 数组
        """
        try:
            # 1. 读取CSV文件
            df = pd.read_csv(data_path)
            
            # 2. 验证数据完整性
            self._validate_input_data(df)
            
            # 3. 处理异常值
            df = self._handle_outliers(df)
            
            # 4. 提取过去4小时的特征数据
            past_hour_data = df[self.pst_hour_feature_names].values[-self.config['lookback_minutes']:]
            
            # 5. 获取当前时间特征
            current_time = pd.to_datetime(df['datetime'].iloc[-1])
            forecast_end = current_time + pd.Timedelta(minutes=self.config['forecast_minutes'])
            cur_datetime = np.array(self._create_time_features(current_time, forecast_end))
            
            # 6. 获取历史模式特征
            dayback_data = self._get_dayback_features(
                df, 
                pd.to_datetime(df['datetime']), 
                current_time, 
                len(df)-1, 
                'nb_computing'
            )
            
            # 7. 特征缩放
            past_hour_scaled = self.feature_scaler.transform(past_hour_data)
            dayback_scaled = self.dayback_scaler.transform(dayback_data.reshape(1, -1))
            
            return past_hour_scaled, cur_datetime, dayback_scaled[0]
            
        except Exception as e:
            raise RuntimeError(f"处理数据记录时出错: {str(e)}")
    