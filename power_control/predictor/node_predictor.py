import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import holidays
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import logging
import random
from pathlib import Path

plt.switch_backend('agg')

# 设置基础目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'power_control', 'predictor', 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'power_control', 'predictor', 'models')
LOG_DIR = os.path.join(BASE_DIR, 'power_control', 'predictor', 'logs')

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 模型配置，使用绝对路径
MODEL_CONFIG = {
    'epochs': 1,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    'lookback_minutes': 360,
    'forecast_minutes': 60,
    'model_dir': MODEL_DIR,  # 修正为绝对路径
    'data_path': os.path.join(DATA_DIR, 'training_data_20250116_161848.csv'),  # 修正为绝对路径
    'log_dir': LOG_DIR,  # 修正为绝对路径
}

class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    def __init__(self, historical_data, time_features, pattern_features, targets):
        self.historical_data = torch.FloatTensor(historical_data)
        self.time_features = torch.FloatTensor(time_features)
        self.pattern_features = torch.FloatTensor(pattern_features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'historical': self.historical_data[idx],
            'time': self.time_features[idx],
            'pattern': self.pattern_features[idx],
            'target': self.targets[idx]
        }

class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.historical_pattern_scaler = MinMaxScaler()
        self.cn_holidays = holidays.CN()
        self.feature_size = 3
        
        # 定义特征列名
        self.historical_feature_names = [
            'running_jobs',    # 运行中的作业数
            'nb_computing',    # 计算节点数
            'utilization_rate' # 利用率
        ]

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
        
        # 初始化数据存储列表
        historical_sequences = []  # 存储历史序列数据
        time_feature_vectors = []  # 存储时间特征
        pattern_feature_vectors = []  # 存储历史模式特征
        target_values = []  # 存储目标值
        
        timestamps = pd.to_datetime(df['datetime'])
        
        # 遍历数据生成训练样本
        for i in range(len(df) - lookback - forecast_horizon + 1):
            # 1. 提取历史序列数据
            historical_data = df[self.historical_feature_names].iloc[i:(i + lookback)].values
            
            # 2. 获取目标时间点
            target_time = timestamps[i + lookback + forecast_horizon - 1]
            
            # 3. 生成时间特征
            time_features = self._create_time_features(target_time)
            
            # 4. 获取历史模式特征
            pattern_features = self._get_historical_patterns(
                df, timestamps, target_time, i + lookback, 'nb_computing'
            )
            
            # 5. 存储数据
            historical_sequences.append(historical_data)
            time_feature_vectors.append(time_features)
            pattern_feature_vectors.append(pattern_features)
            target_values.append(df['nb_computing'].iloc[i + lookback + forecast_horizon - 1])
        
        return (np.array(historical_sequences),
                np.array(time_feature_vectors),
                np.array(pattern_feature_vectors),
                np.array(target_values).reshape(-1, 1))

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

    def _get_historical_patterns(self, df, timestamps, target_time, current_idx, target_col):
        """
        获取历史模式特征
        
        参数:
            df (pd.DataFrame): 数据框
            timestamps (pd.Series): 时间戳序列
            target_time (datetime): 目标时间点
            current_idx (int): 当前索引
            target_col (str): 目标列名
            
        返回:
            np.array: 历史模式特征
        """
        pattern_features = []
        
        # 获取不同时间跨度的历史数据
        for minutes_back in [24*60, 7*24*60, 14*24*60]:  # 1天、7天、14天
            historical_idx = current_idx - minutes_back
            if historical_idx >= 0:
                pattern_features.append(df[target_col].iloc[historical_idx])
            else:
                pattern_features.append(0)
        
        return np.array(pattern_features)

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
            # 创建数据集
            dataset = TimeSeriesDataset(
                data_dict[f'X_{split}'][0],  # historical data
                data_dict[f'X_{split}'][1],  # time features
                data_dict[f'X_{split}'][2],  # pattern features
                data_dict[f'y_{split}']
            )
            
            # 创建数据加载器
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=4,
                pin_memory=True
            )
        
        return loaders

    def load_and_prepare_data(self, data: pd.DataFrame) -> dict:
        """
        加载并准备模型训练所需的数据
        
        参数:
            data (pd.DataFrame): 输入数据框
            
        返回:
            dict: 包含处理后的训练、验证和测试数据的字典
        """
        # 确保数据按时间排序
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # 准备时间序列数据
        historical_sequences, time_features, pattern_features, targets = \
            self.prepare_time_series_data(data)
        
        print(f"历史序列: {historical_sequences.shape}")
        print(f"时间特征: {time_features.shape}")
        print(f"模式特征: {pattern_features.shape}")
        print(f"目标值: {targets.shape}")
        
        # 划分数据集
        train_size = int(len(targets) * 0.7)
        val_size = int(len(targets) * 0.15)
        
        # 训练集
        X_train = [
            historical_sequences[:train_size],
            time_features[:train_size],
            pattern_features[:train_size]
        ]
        y_train = targets[:train_size]
        
        # 验证集
        X_val = [
            historical_sequences[train_size:train_size + val_size],
            time_features[train_size:train_size + val_size],
            pattern_features[train_size:train_size + val_size]
        ]
        y_val = targets[train_size:train_size + val_size]
        
        # 测试集
        X_test = [
            historical_sequences[train_size + val_size:],
            time_features[train_size + val_size:],
            pattern_features[train_size + val_size:]
        ]
        y_test = targets[train_size + val_size:]

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
        X_train[2] = self.historical_pattern_scaler.fit_transform(X_train[2])
        X_val[2] = self.historical_pattern_scaler.transform(X_val[2])
        X_test[2] = self.historical_pattern_scaler.transform(X_test[2])
        
        # 目标值缩放
        y_train = self.target_scaler.fit_transform(y_train)
        y_val = self.target_scaler.transform(y_val)
        y_test = self.target_scaler.transform(y_test)

        print('训练集长度', len(X_train[0]))
        print('验证集长度', len(X_val[0]))
        print('测试集长度', len(X_test[0]))
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
        return self.target_scaler.inverse_transform(y_scaled)
    
    # 定义缺失的方法
    def prepare_historical_data(self, current_data: pd.DataFrame):
        """准备历史数据特征"""
        return current_data[self.historical_feature_names].values

    def prepare_time_features(self, target_time: datetime):
        """准备时间特征"""
        return np.array(self._create_time_features(target_time))

    def prepare_pattern_features(self, current_data: pd.DataFrame, target_time: datetime):
        """准备历史模式特征"""
        current_idx = len(current_data) - 1
        return self._get_historical_patterns(
            current_data, pd.to_datetime(current_data['datetime']),
            target_time, current_idx, 'nb_computing'
        )

class AttentionLayer(nn.Module):
    """注意力层"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # 加权求和
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class NodePredictorNN(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        
        # LSTM配置
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 注意力层
        self.attention = AttentionLayer(self.lstm_hidden_size * 2)  # 双向LSTM输出维度翻倍
        
        # 时间特征处理 (6个时间特征)
        self.time_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        
        # 历史同期数据处理 (3个历史同期特征)
        self.pattern_fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2)
        )
        
        # 计算组合特征的总维度
        # LSTM输出: lstm_hidden_size * 2 (双向) = 256
        # 时间特征: 32
        # 历史同期特征: 16
        combined_input_size = 256 + 32 + 16  # 总共304维
        
        # 组合特征的全连接层
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_historical, x_time, x_patterns):
        batch_size = x_historical.size(0)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x_historical)
        
        # 应用注意力机制
        lstm_out, attention_weights = self.attention(lstm_out)
        
        # 处理时间特征
        time_out = self.time_fc(x_time)
        
        # 处理历史同期数据
        pattern_out = self.pattern_fc(x_patterns)
        
        # 组合所有特征
        combined = torch.cat([lstm_out, time_out, pattern_out], dim=1)
        
        return self.combined_fc(combined)

def smape(y_true, y_pred):
    """计算SMAPE"""
    return 100 * np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))

class NodePredictor:
    def __init__(self, config: dict = None):
        self.config = config or MODEL_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_processor = DataProcessor(self.config)
        self.model = None
        self.loaded_model_path = None  # 初始化为None
        
        # 创建必要的目录
        self._setup_directories()
        
        # 设置日志记录器
        self.logger = logging.getLogger(self.timestamp)
        self.logger.setLevel(logging.INFO)
        
        # 防止重复添加处理器
        if not self.logger.handlers:
            # 创建文件处理器
            fh = logging.FileHandler(self.train_log_file)
            fh.setLevel(logging.INFO)
            
            # 创建控制台处理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # 创建日志格式
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            # 添加处理器到日志器
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        
    def _setup_directories(self):
        """创建必要的目录结构"""
        # 创建运行时间戳
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 创建本次运行的日志根目录
        self.run_dir = os.path.join(self.config['log_dir'], self.timestamp)
        
        # 创建各类日志的子目录
        self.train_log_dir = os.path.join(self.run_dir, 'training')
        self.eval_log_dir = os.path.join(self.run_dir, 'evaluation')
        self.model_save_dir = os.path.join(self.config['model_dir'], self.timestamp)
        
        # 创建所有必要的目录
        for directory in [self.train_log_dir, self.eval_log_dir, self.model_save_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 设置日志文件路径
        self.train_log_file = os.path.join(self.train_log_dir, 'training.log')
        self.eval_log_file = os.path.join(self.eval_log_dir, 'evaluation.log')
    
    def _calculate_smape(self, y_true, y_pred, epsilon=1e-10):
        """
        计算对称平均绝对百分比误差 (SMAPE)，添加epsilon避免除零错误
        
        参数:
            y_true (np.ndarray): 真实值
            y_pred (np.ndarray): 预测值
            epsilon (float): 小常数，防止除零
            
        返回:
            float: SMAPE值（百分比）
        """
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_pred) + np.abs(y_true)) / 2 + epsilon
        return 100 * np.mean(numerator / denominator)

    def _plot_training_history(self, history, save_path):
        """
        绘制训练历史曲线
        
        参数:
            history (dict): 包含训练历史的字典
            save_path (str): 图表保存路径
        """
        plt.figure(figsize=(12, 8))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'training_history.png'))
        plt.close()

    def _plot_error_distribution(self, errors, save_path):
        """
        绘制预测误差分布
        
        参数:
            errors (np.ndarray): 预测误差数组
            save_path (str): 图表保存路径
        """
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error (nodes)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'error_distribution.png'))
        plt.close()

    def _plot_prediction_scatter(self, targets, predictions, save_dir):
        """绘制预测散点图"""
        plt.figure(figsize=(10, 10))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'))
        plt.close()

    def _plot_time_series(self, targets, predictions, save_dir):
        """绘制时间序列对比图"""
        plt.figure(figsize=(15, 6))
        plt.plot(targets, label='Actual', alpha=0.7)
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.title('Time Series Prediction')
        plt.xlabel('Time Steps')
        plt.ylabel('Number of Nodes')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'time_series.png'))
        plt.close()

    def train(self, data_dict: dict):
        """训练模型并记录训练过程"""
        self.logger.info(f"开始在设备 {self.device} 上训练模型")
        
        data_loaders = self.data_processor.create_data_loaders(
            data_dict,
            self.config['batch_size']
        )
        
        self.model = NodePredictorNN(
            feature_size=self.data_processor.feature_size
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-5
        )
        
        # 修正：从 torch.optim.lr_scheduler 导入 ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 记录训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        for epoch in range(1, self.config['epochs'] + 1):
            # 训练阶段
            self.model.train()
            train_loss = 0
            batch_count = len(data_loaders['train'])
            total_epochs = self.config['epochs']
            print(f"\n正在训练 Epoch {epoch}/{self.config['epochs']}")
            print("进度: ", end="")
        
            for batch_idx, batch in enumerate(data_loaders['train'], 1):
                optimizer.zero_grad()
                
                historical = batch['historical'].to(self.device)
                time_feat = batch['time'].to(self.device)
                pattern = batch['pattern'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(historical, time_feat, pattern)
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()

                progress = int(50 * batch_idx / batch_count)
                print(f"\r进度: [{'=' * progress}{' ' * (50-progress)}] {batch_idx}/{batch_count} "
                    f"- 当前 loss: {loss.item():.6f}", end="")
            
            train_loss /= len(data_loaders['train'])
            
            # 验证阶段
            val_loss = self._validate(data_loaders['val'], criterion)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # 更新学习率
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch}/{total_epochs}:")
            print(f"训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"学习率: {current_lr:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(optimizer, scheduler)
                self.logger.info(f"Epoch {epoch}: Val Loss improved to {val_loss:.6f}. 模型已保存。")
            else:
                patience_counter += 1
                self.logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.6f} 不改善。")
                if patience_counter >= self.config['early_stopping_patience']:
                    self.logger.info(f"早停触发于 epoch {epoch}")
                    break
            
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # 绘制训练历史
        self._plot_training_history(history, self.train_log_dir)
        
        # 保存训练历史
        history_path = os.path.join(self.train_log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        self.logger.info("训练完成。训练历史已保存。")
        
        return history

    def evaluate(self, data_dict: dict) -> tuple:
        """评估模型并生成可视化结果"""
        self.logger.info("开始评估模型")
        
        self.model.eval()
        
        # 获取测试数据加载器
        test_loader = DataLoader(
            TimeSeriesDataset(
                data_dict['X_test'][0],
                data_dict['X_test'][1],
                data_dict['X_test'][2],
                data_dict['y_test']
            ),
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 将数据移到设备上
                historical = batch['historical'].to(self.device)
                time_feat = batch['time'].to(self.device)
                pattern = batch['pattern'].to(self.device)
                
                # 获取预测结果
                outputs = self.model(historical, time_feat, pattern)
                predictions.append(outputs.cpu().numpy())
                targets.append(batch['target'].numpy())
        
        # 合并批次结果
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # 转换回原始尺度
        predictions = self.data_processor.inverse_transform_y(predictions)
        targets = self.data_processor.inverse_transform_y(targets)
        
        # 计算评估指标
        metrics = {
            'mse': float(mean_squared_error(targets, predictions)),  # 转换为Python float
            'mae': float(mean_absolute_error(targets, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(targets, predictions))),
            'r2': float(r2_score(targets, predictions)),
            'smape': float(self._calculate_smape(targets, predictions))
        }
        
        # 计算不同容差下的准确率
        predictions_discrete = np.maximum(0, np.round(predictions))
        targets_discrete = np.maximum(0, np.round(targets))
        
        for tolerance in [0, 1, 2, 5]:
            if tolerance == 0:
                key = 'accuracy_exact'
            else:
                key = f'accuracy_within_{tolerance}'
            within_tolerance = np.abs(predictions_discrete - targets_discrete) <= tolerance
            metrics[key] = float(np.mean(within_tolerance) * 100)  # 转换为Python float
        
        # 添加误差分布统计
        metrics['error_distribution'] = self._convert_to_python_types(
            self._calculate_error_distribution(targets_discrete, predictions_discrete)
        )
        
        # 保存评估指标
        metrics_path = os.path.join(self.eval_log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 生成可视化结果
        self._plot_error_distribution(predictions - targets, self.eval_log_dir)
        self._plot_prediction_scatter(targets, predictions, self.eval_log_dir)
        self._plot_time_series(targets, predictions, self.eval_log_dir)
        
        self.logger.info("评估完成。评估结果已保存。")
        
        return metrics, predictions, targets

    def _convert_to_python_types(self, d: dict) -> dict:
        """将字典中的NumPy类型转换为Python原生类型"""
        converted = {}
        for k, v in d.items():
            if isinstance(v, (np.int32, np.int64)):
                converted[k] = int(v)
            elif isinstance(v, (np.float32, np.float64)):
                converted[k] = float(v)
            elif isinstance(v, dict):
                converted[k] = self._convert_to_python_types(v)
            elif isinstance(v, (list, tuple)):
                converted[k] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in v]
            else:
                converted[k] = v
        return converted

    def predict(self, historical_data: np.ndarray, time_features: np.ndarray, 
                pattern_features: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        参数:
            historical_data (np.ndarray): 历史数据特征，形状为 (batch_size, lookback, feature_size)
            time_features (np.ndarray): 时间特征，形状为 (batch_size, time_feature_size)
            pattern_features (np.ndarray): 历史模式特征，形状为 (batch_size, pattern_feature_size)
            
        返回:
            np.ndarray: 预测结果
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        
        # 确保模型处于评估模式
        self.model.eval()
        
        # 检查输入维度
        if len(historical_data.shape) != 3:
            raise ValueError(f"Expected historical_data to have 3 dimensions, "
                           f"got shape {historical_data.shape}")
        if len(time_features.shape) != 2:
            raise ValueError(f"Expected time_features to have 2 dimensions, "
                           f"got shape {time_features.shape}")
        if len(pattern_features.shape) != 2:
            raise ValueError(f"Expected pattern_features to have 2 dimensions, "
                           f"got shape {pattern_features.shape}")
        
        # 创建数据加载器
        predict_dataset = TimeSeriesDataset(
            historical_data,
            time_features,
            pattern_features,
            np.zeros((len(historical_data), 1))  # 虚拟目标值
        )
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        predictions = []
        
        # 进行预测
        with torch.no_grad():
            for batch in predict_loader:
                historical = batch['historical'].to(self.device)
                time_feat = batch['time'].to(self.device)
                pattern = batch['pattern'].to(self.device)
                
                outputs = self.model(historical, time_feat, pattern)
                predictions.append(outputs.cpu().numpy())
        
        # 合并批次结果
        predictions = np.concatenate(predictions)
        
        # 转换回原始尺度
        predictions = self.data_processor.inverse_transform_y(predictions)
        
        return predictions

    def predict_next(self, current_data: pd.DataFrame, target_time: datetime) -> int:
        """
        预测下一个时间点的节点数
        
        参数:
            current_data (pd.DataFrame): 当前时间窗口的数据
            target_time (datetime): 目标预测时间点
            
        返回:
            int: 预测的节点数（四舍五入到整数）
        """
        # 准备特征
        historical_data = self.data_processor.prepare_historical_data(current_data)
        time_features = self.data_processor.prepare_time_features(target_time)
        pattern_features = self.data_processor.prepare_pattern_features(
            current_data, target_time
        )
        
        # 扩展维度以匹配批处理格式
        historical_data = np.expand_dims(historical_data, axis=0)
        time_features = np.expand_dims(time_features, axis=0)
        pattern_features = np.expand_dims(pattern_features, axis=0)
        
        # 进行预测
        prediction = self.predict(historical_data, time_features, pattern_features)
        
        # 返回四舍五入后的预测值
        return int(round(float(prediction[0])))

    def predict_sequence(self, start_data: pd.DataFrame, 
                        forecast_steps: int) -> np.ndarray:
        """
        预测未来多个时间步的节点数序列
        
        参数:
            start_data (pd.DataFrame): 起始时间窗口的数据
            forecast_steps (int): 预测步数
            
        返回:
            np.ndarray: 预测的节点数序列
        """
        predictions = []
        current_data = start_data.copy()
        current_time = pd.to_datetime(current_data['datetime'].iloc[-1])
        
        for _ in range(forecast_steps):
            # 更新目标时间，根据 forecast_minutes
            current_time += pd.Timedelta(minutes=self.config['forecast_minutes'])
            
            # 预测下一个时间点
            next_value = self.predict_next(current_data, current_time)
            predictions.append(next_value)
            
            # 更新数据框，添加预测值
            new_row = pd.DataFrame({
                'datetime': [current_time],
                'nb_computing': [next_value],
                'running_jobs': [current_data['running_jobs'].iloc[-1]],  # 使用最后一个值
                'utilization_rate': [current_data['utilization_rate'].iloc[-1]]  # 使用最后一个值
            })
            
            # 移除最早的一行，添加新预测的一行
            current_data = pd.concat([current_data.iloc[1:], new_row])
            current_data.reset_index(drop=True, inplace=True)
        
        return np.array(predictions)

    def save_model(self, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None, path: str = None):
        """
        保存模型、优化器和调度器的状态及配置
        
        参数:
            optimizer (torch.optim.Optimizer, optional): 优化器实例。
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 调度器实例。
            path (str, optional): 保存路径。如果未指定，将使用默认路径。
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if path is None:
            # 使用默认路径
            path = os.path.join(self.model_save_dir, 'checkpoint.pth')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            # 创建检查点
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            # 保存检查点
            torch.save(checkpoint, path)
            
            # 保存配置
            config_path = os.path.join(os.path.dirname(path), 'config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            self.logger.info(f"模型、优化器和调度器状态已保存到 {os.path.dirname(path)}")
            
        except Exception as e:
            raise RuntimeError(f"保存模型到 {path} 时出错: {str(e)}")

    def load_model(self, path: str = None, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None):
        """
        加载预训练模型及其优化器和调度器状态
        
        参数:
            path (str, optional): 模型文件路径。如果未指定，将加载最新的模型。
            optimizer (torch.optim.Optimizer, optional): 优化器实例。
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 调度器实例。
        """
        if path is None:
            # 如果未指定路径，查找最新的模型
            model_dir = self.config['model_dir']
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"模型目录未找到: {model_dir}")
            
            # 获取所有模型子目录（按时间戳命名）
            subdirs = [d for d in os.listdir(model_dir) 
                      if os.path.isdir(os.path.join(model_dir, d))]
            
            if not subdirs:
                raise FileNotFoundError(f"{model_dir} 中未找到模型检查点")
            
            # 选择最新的模型目录
            latest_dir = max(subdirs)
            path = os.path.join(model_dir, latest_dir, 'checkpoint.pth')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件未找到: {path}")
        
        # 加载模型配置
        config_path = os.path.join(os.path.dirname(path), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
                # 更新配置，但保留当前的目录设置
                saved_config.update({
                    'model_dir': self.config['model_dir'],
                    'log_dir': self.config['log_dir'],
                    'data_path': self.config['data_path']
                })
                self.config = saved_config
                # 重新初始化 DataProcessor
                self.data_processor = DataProcessor(self.config)
        else:
            self.logger.warning(f"配置文件未找到于 {config_path}。使用现有配置。")
        
        # 创建新的模型实例
        self.model = NodePredictorNN(
            feature_size=self.data_processor.feature_size
        ).to(self.device)
        
        try:
            # 加载检查点
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"成功从 {path} 加载模型")
            
            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.logger.info("成功加载优化器状态")
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("成功加载调度器状态")
            
            # 记录加载的模型路径
            self.loaded_model_path = path
            
        except Exception as e:
            raise RuntimeError(f"从 {path} 加载模型时出错: {str(e)}")
        
        # 设置为评估模式
        self.model.eval()

    def get_model_info(self) -> dict:
        """
        获取当前加载的模型信息
        
        返回:
            dict: 包含模型信息的字典
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "status": "Model loaded",
            "model_path": getattr(self, 'loaded_model_path', 'Unknown'),
            "device": str(self.device),
            "feature_size": self.data_processor.feature_size,
            "config": self.config
        }
        
        # 获取模型参数数量
        total_params = sum(
            p.numel() for p in self.model.parameters()
        )
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        
        info.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        })
        
        # 如果模型是从文件加载的，添加文件信息
        if hasattr(self, 'loaded_model_path'):
            model_file = Path(self.loaded_model_path)
            info.update({
                "model_size_MB": model_file.stat().st_size / (1024 * 1024),  # MB
                "last_modified": datetime.fromtimestamp(
                    model_file.stat().st_mtime
                ).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return info

    def evaluate_with_cv(self, data_dict: dict) -> dict:
        """
        使用时间序列交叉验证评估模型
        
        参数:
            data_dict (dict): 包含训练数据的字典
            
        返回:
            dict: 交叉验证的评估指标
        """
        self.logger.info("开始时间序列交叉验证评估")
        
        # 准备数据
        X_hist = data_dict['X_train'][0]
        X_time = data_dict['X_train'][1]
        X_pattern = data_dict['X_train'][2]
        y = data_dict['y_train']
        
        n_splits = 5  # 交叉验证折数
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 存储每折的评估指标
        cv_metrics = {
            'mse': [], 'mae': [], 'rmse': [], 'r2': [], 'smape': [],
            'accuracy_exact': [], 'accuracy_within_1': [],
            'accuracy_within_2': [], 'accuracy_within_5': []
        }
        
        # 对每一折进行训练和评估
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_hist), 1):
            self.logger.info(f"\n训练第 {fold}/{n_splits} 折")
            
            # 准备当前折的数据
            X_train_hist = X_hist[train_idx]
            X_train_time = X_time[train_idx]
            X_train_pattern = X_pattern[train_idx]
            y_train = y[train_idx]
            
            X_val_hist = X_hist[val_idx]
            X_val_time = X_time[val_idx]
            X_val_pattern = X_pattern[val_idx]
            y_val = y[val_idx]
            
            # 创建数据加载器
            train_dataset = TimeSeriesDataset(
                X_train_hist, X_train_time, X_train_pattern, y_train
            )
            val_dataset = TimeSeriesDataset(
                X_val_hist, X_val_time, X_val_pattern, y_val
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # 创建新的模型实例
            fold_model = NodePredictorNN(
                feature_size=self.data_processor.feature_size
            ).to(self.device)
            
            # 创建优化器和损失函数
            optimizer = torch.optim.AdamW(
                fold_model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=1e-5
            )
            criterion = nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            
            # 训练当前折的模型
            best_val_loss = float('inf')
            patience_counter = 0
            best_fold_model_state = None
            
            for epoch in range(1, self.config['epochs'] + 1):
                # 训练阶段
                fold_model.train()
                train_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    historical = batch['historical'].to(self.device)
                    time_feat = batch['time'].to(self.device)
                    pattern = batch['pattern'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    outputs = fold_model(historical, time_feat, pattern)
                    loss = criterion(outputs, targets)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(fold_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # 验证阶段
                fold_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        historical = batch['historical'].to(self.device)
                        time_feat = batch['time'].to(self.device)
                        pattern = batch['pattern'].to(self.device)
                        targets = batch['target'].to(self.device)
                        
                        outputs = fold_model(historical, time_feat, pattern)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_fold_model_state = fold_model.state_dict().copy()
                    self.logger.info(f"折 {fold}  Epoch {epoch}: Val Loss 改善到 {val_loss:.6f}")
                else:
                    patience_counter += 1
                    self.logger.info(f"折 {fold}  Epoch {epoch}: Val Loss = {val_loss:.6f} 不改善")
                    if patience_counter >= self.config['early_stopping_patience']:
                        self.logger.info(f"折 {fold} 早停触发于 epoch {epoch}")
                        break
            
            # 加载最佳模型状态
            if best_fold_model_state:
                fold_model.load_state_dict(best_fold_model_state)
            
            # 在验证集上评估
            fold_model.eval()
            fold_predictions = []
            fold_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    historical = batch['historical'].to(self.device)
                    time_feat = batch['time'].to(self.device)
                    pattern = batch['pattern'].to(self.device)
                    
                    outputs = fold_model(historical, time_feat, pattern)
                    fold_predictions.append(outputs.cpu().numpy())
                    fold_targets.append(batch['target'].numpy())
            
            fold_predictions = np.concatenate(fold_predictions)
            fold_targets = np.concatenate(fold_targets)
            
            # 转换回原始尺度
            fold_predictions = self.data_processor.inverse_transform_y(fold_predictions)
            fold_targets = self.data_processor.inverse_transform_y(fold_targets)
            
            # 计算评估指标
            fold_metrics = {
                'mse': mean_squared_error(fold_targets, fold_predictions),
                'mae': mean_absolute_error(fold_targets, fold_predictions),
                'rmse': np.sqrt(mean_squared_error(fold_targets, fold_predictions)),
                'r2': r2_score(fold_targets, fold_predictions),
                'smape': self._calculate_smape(fold_targets, fold_predictions)
            }
            
            # 计算不同容差下的准确率
            fold_predictions_discrete = np.maximum(0, np.round(fold_predictions))
            fold_targets_discrete = np.maximum(0, np.round(fold_targets))
            
            for tolerance in [0, 1, 2, 5]:
                if tolerance == 0:
                    key = 'accuracy_exact'
                else:
                    key = f'accuracy_within_{tolerance}'
                within_tolerance = np.abs(fold_predictions_discrete - fold_targets_discrete) <= tolerance
                fold_metrics[key] = np.mean(within_tolerance) * 100
            
            # 存储当前折的指标
            for metric in cv_metrics:
                cv_metrics[metric].append(fold_metrics[metric])
            
            self.logger.info(f"\n折 {fold} 结果:")
            for metric, value in fold_metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
        
        # 计算平均指标
        mean_metrics = {
            metric: np.mean(values) for metric, values in cv_metrics.items()
        }
        std_metrics = {
            metric: np.std(values) for metric, values in cv_metrics.items()
        }
        
        self.logger.info("\n时间序列交叉验证结果:")
        for metric in mean_metrics:
            self.logger.info(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
        
        return {
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'fold_metrics': cv_metrics
        }

    def _validate(self, data_loader, criterion):
        """验证模型性能"""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        for batch in data_loader:
            # 将数据移到设备上
            historical = batch['historical'].to(self.device)
            time_feat = batch['time'].to(self.device)
            pattern = batch['pattern'].to(self.device)
            targets = batch['target'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(historical, time_feat, pattern)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else float('inf')

    def _calculate_error_distribution(self, y_true, y_pred):
        """
        计算预测误差的详细分布
        
        参数:
            y_true (np.ndarray): 真实值
            y_pred (np.ndarray): 预测值
            
        返回:
            dict: 包含各种误差统计指标的字典
        """
        # 计算误差
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        # 计算基本统计量
        distribution = {
            'mean_error': float(np.mean(errors)),  # 平均误差（可能为负）
            'mean_abs_error': float(np.mean(abs_errors)),  # 平均绝对误差
            'median_error': float(np.median(errors)),  # 中位数误差
            'max_error': float(np.max(abs_errors)),  # 最大绝对误差
            'error_std': float(np.std(errors)),  # 误差标准差
            
            # 误差分位数
            'error_25th': float(np.percentile(errors, 25)),
            'error_50th': float(np.percentile(errors, 50)),
            'error_75th': float(np.percentile(errors, 75)),
            'error_90th': float(np.percentile(errors, 90)),
            'error_95th': float(np.percentile(errors, 95)),
            
            # 误差范围内的预测比例
            'within_1_node': float(np.mean(abs_errors <= 1) * 100),
            'within_2_nodes': float(np.mean(abs_errors <= 2) * 100),
            'within_5_nodes': float(np.mean(abs_errors <= 5) * 100),
            'within_10_nodes': float(np.mean(abs_errors <= 10) * 100),
            
            # 过预测和欠预测的比例
            'over_prediction': float(np.mean(errors > 0) * 100),
            'under_prediction': float(np.mean(errors < 0) * 100),
            'exact_prediction': float(np.mean(errors == 0) * 100)
        }
        
        # 计算误差区间分布
        error_ranges = [(0, 1), (1, 2), (2, 5), (5, 10), (10, float('inf'))]
        for start, end in error_ranges:
            key = f'error_{start}_to_{end}'
            if end == float('inf'):
                distribution[key] = float(np.mean(abs_errors > start) * 100)
            else:
                distribution[key] = float(np.mean((abs_errors > start) & 
                                                (abs_errors <= end)) * 100)
        
        return distribution

    def _plot_prediction_scatter(self, targets, predictions, save_dir):
        """绘制预测散点图"""
        plt.figure(figsize=(10, 10))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.title('Predicted vs Actual Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'))
        plt.close()

    def _plot_time_series(self, targets, predictions, save_dir):
        """绘制时间序列对比图"""
        plt.figure(figsize=(15, 6))
        plt.plot(targets, label='Actual', alpha=0.7)
        plt.plot(predictions, label='Predicted', alpha=0.7)
        plt.title('Time Series Prediction')
        plt.xlabel('Time Steps')
        plt.ylabel('Number of Nodes')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'time_series.png'))
        plt.close()

def set_seed(seed: int = 42):
    """
    设置随机种子以确保结果可重现
    
    参数:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 某些操作的确定性配置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # 创建预测器实例
    predictor = NodePredictor()
    
    # 加载原始数据
    data = pd.read_csv(predictor.config['data_path'])
    
    # 准备数据
    print("\nPreparing data...")
    predictor.logger.info("开始准备数据")
    data_dict = predictor.data_processor.load_and_prepare_data(data)
    print(len(data_dict['X_train'][0]))

    # 训练模型
    print("\nTraining model...")
    predictor.logger.info("开始训练模型")
    history = predictor.train(data_dict)
    
    # 评估模型
    print("\nEvaluating model...")
    predictor.logger.info("开始评估模型")
    test_metrics, predictions, targets = predictor.evaluate(data_dict)
    
    # 打印测试集结果
    print("\n=== Test Set Results ===")
    print("\nBasic Metrics:")
    print(f"Mean Squared Error: {test_metrics['mse']:.4f}")
    print(f"Root Mean Squared Error: {test_metrics['rmse']:.4f}")
    print(f"Mean Absolute Error: {test_metrics['mae']:.4f}")
    print(f"R² Score: {test_metrics['r2']:.4f}")
    print(f"SMAPE: {test_metrics['smape']:.2f}%")
    
    print("\nAccuracy Metrics:")
    print(f"Exact Match: {test_metrics['accuracy_exact']:.2f}%")
    print(f"Within ±1 node: {test_metrics['accuracy_within_1']:.2f}%")
    print(f"Within ±2 nodes: {test_metrics['accuracy_within_2']:.2f}%")
    print(f"Within ±5 nodes: {test_metrics['accuracy_within_5']:.2f}%")
    
    # 打印误差分布
    dist = test_metrics['error_distribution']
    print("\nError Distribution:")
    print(f"Mean Error: {dist['mean_error']:.2f} nodes")
    print(f"Mean Absolute Error: {dist['mean_abs_error']:.2f} nodes")
    print(f"Median Error: {dist['median_error']:.2f} nodes")
    print(f"Error Std: {dist['error_std']:.2f} nodes")
    print(f"Max Error: {dist['max_error']:.2f} nodes")
    
    print("\nError Percentiles:")
    print(f"25th percentile: {dist['error_25th']:.2f} nodes")
    print(f"50th percentile: {dist['error_50th']:.2f} nodes")
    print(f"75th percentile: {dist['error_75th']:.2f} nodes")
    print(f"90th percentile: {dist['error_90th']:.2f} nodes")
    print(f"95th percentile: {dist['error_95th']:.2f} nodes")
    
    print("\nPrediction Bias:")
    print(f"Exact Predictions: {dist['exact_prediction']:.2f}%")
    print(f"Over-predictions: {dist['over_prediction']:.2f}%")
    print(f"Under-predictions: {dist['under_prediction']:.2f}%")
    
    print("\nError Range Distribution:")
    print(f"0-1 nodes: {dist['error_0_to_1']:.2f}%")
    print(f"1-2 nodes: {dist['error_1_to_2']:.2f}%")
    print(f"2-5 nodes: {dist['error_2_to_5']:.2f}%")
    print(f"5-10 nodes: {dist['error_5_to_10']:.2f}%")
    print(f">10 nodes: {dist['error_10_to_inf']:.2f}%")
    
    # 保存实验结果摘要
    summary_path = os.path.join(predictor.run_dir, 'experiment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== Experiment Summary ===\n\n")
        f.write(f"Timestamp: {predictor.timestamp}\n")
        f.write(f"Device: {predictor.device}\n\n")
        f.write("Model Configuration:\n")
        f.write(json.dumps(predictor.config, indent=2))
        f.write("\n\nTest Set Results:\n")
        f.write(f"MSE: {test_metrics['mse']:.4f}\n")
        f.write(f"RMSE: {test_metrics['rmse']:.4f}\n")
        f.write(f"MAE: {test_metrics['mae']:.4f}\n")
        f.write(f"R²: {test_metrics['r2']:.4f}\n")
        f.write(f"SMAPE: {test_metrics['smape']:.2f}%\n")
    
    predictor.logger.info("实验完成。实验摘要已保存。")

if __name__ == '__main__':
    main()
