from config import MODEL_CONFIG
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader
import torch
from model import NodePredictorNN
from data_processor import DataProcessor, TimeSeriesDataset

class Predictor:
    def __init__(self):
        pass

    def predict(self, past_hour_features: np.ndarray, cur_datetime_features: np.ndarray, 
                dayback_features: np.ndarray) -> np.ndarray:
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
        if len(past_hour_features.shape) != 3:
            raise ValueError(f"Expected historical_data to have 3 dimensions, "
                           f"got shape {past_hour_features.shape}")
        if len(cur_datetime_features.shape) != 2:
            raise ValueError(f"Expected time_features to have 2 dimensions, "
                           f"got shape {cur_datetime_features.shape}")
        if len(dayback_features.shape) != 2:
            raise ValueError(f"Expected pattern_features to have 2 dimensions, "
                           f"got shape {dayback_features.shape}")
        
        # 创建数据加载器
        predict_dataset = TimeSeriesDataset(
            past_hour_features,
            cur_datetime_features,
            dayback_features,
            np.zeros((len(past_hour_features), 1))  # 虚拟目标值
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
                past_hour = batch['past_hour'].to(self.device)
                cur_datetime = batch['cur_datetime'].to(self.device)
                dayback = batch['dayback'].to(self.device)
                
                outputs = self.model(past_hour, cur_datetime, dayback)
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