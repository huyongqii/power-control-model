import os
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from config import MODEL_CONFIG
import pandas as pd

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

class DataLoader:
    def __init__(self):
        self.config = MODEL_CONFIG
        data_filename = os.path.splitext(os.path.basename(self.config['data_path']))[0]
        self.dataset_dir = os.path.join(os.path.dirname(self.config['data_path']), data_filename)  
        self._load_scalers()
        self.feature_size = 6
    
    def _load_scalers(self):
        """加载数据缩放器"""
        scaler_path = os.path.join(self.dataset_dir, "dataset_scalers.pkl")
        scalers = joblib.load(scaler_path)
        self.feature_scaler = scalers['feature_scaler']
        self.target_scaler = scalers['target_scaler']
        self.dayback_scaler = scalers['dayback_scaler']
    
    def load_data(self, split: str = 'all'):
        """
        加载指定的数据集划分
        
        参数:
            split (str): 要加载的数据集划分('train', 'val', 'test', 'all')
            
        返回:
            dict: 包含加载的数据集的字典
        """
        try:
            data_dict = {}
            splits = [split] if split != 'all' else ['train', 'val', 'test']
            
            for current_split in splits:
                # 加载特征数据
                X_key = f'X_{current_split}'
                X_data = []
                for i in range(3):
                    filename = f"dataset_{X_key}_part{i}.npy"
                    filepath = os.path.join(self.dataset_dir, filename)
                    X_data.append(np.load(filepath))
                data_dict[X_key] = X_data
                
                # 加载目标值
                y_key = f'y_{current_split}'
                filename = f"dataset_{y_key}.npy"
                filepath = os.path.join(self.dataset_dir, filename)
                data_dict[y_key] = np.load(filepath)
            
            print(f"成功加载{split}数据集")
            return data_dict
            
        except Exception as e:
            raise RuntimeError(f"加载数据集时出错: {str(e)}")
    
    def create_data_loaders(self, batch_size: int, split: str = 'all') -> dict:
        """
        创建数据加载器
        
        参数:
            batch_size (int): 批次大小
            split (str): 要加载的数据集划分
            
        返回:
            dict: 包含数据加载器的字典
        """
        data_dict = self.load_data(split)
        loaders = {}
        
        splits = [split] if split != 'all' else ['train', 'val', 'test']
        for current_split in splits:
            print(f"\n{current_split} 数据集统计:")
            for i, X_part in enumerate(data_dict[f'X_{current_split}']):
                print(f"特征部分 {i}:")
                print(f"范围: [{X_part.min():.6f}, {X_part.max():.6f}]")
                print(f"均值: {X_part.mean():.6f}")
                print(f"标准差: {X_part.std():.6f}")
                # 添加四分位数统计
                q1, q2, q3 = np.percentile(X_part, [25, 50, 75])
                print(f"四分位数: Q1={q1:.6f}, Q2={q2:.6f}, Q3={q3:.6f}")
            
            y = data_dict[f'y_{current_split}']
            print(f"目标值:")
            print(f"范围: [{y.min():.6f}, {y.max():.6f}]")
            print(f"均值: {y.mean():.6f}")
            print(f"标准差: {y.std():.6f}")
            
            dataset = TimeSeriesDataset(
                data_dict[f'X_{current_split}'][0],
                data_dict[f'X_{current_split}'][1],
                data_dict[f'X_{current_split}'][2],
                data_dict[f'y_{current_split}']
            )
            
            loaders[current_split] = TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(current_split == 'train'),
                num_workers=4,
                pin_memory=True
            )
        
        return loaders
    
    def inverse_transform_y(self, y_scaled):
        """将缩放后的目标值转换回原始范围"""
        # 确保输入是2D数组
        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        
        # 进行反向转换
        y_original = self.target_scaler.inverse_transform(y_scaled)
        
        # 转回1D数组
        return y_original.reshape(-1)

    def load_test_data(self):
        """加载整个数据集作为测试集"""
        try:
            data_dict = {}
            data_filename = os.path.splitext(os.path.basename(self.config['data_path']))[0]
            
            # 加载特征数据
            X_data = []
            for i in range(3):
                filename = f"dataset_X_test_part{i}.npy"
                filepath = os.path.join(self.dataset_dir, filename)
                X_data.append(np.load(filepath))
            data_dict['X_test'] = X_data
            
            # 加载目标值
            filename = f"dataset_y_test.npy"
            filepath = os.path.join(self.dataset_dir, filename)
            data_dict['y_test'] = np.load(filepath)
            
            print(f"成功加载测试数据集")
            return data_dict
            
        except Exception as e:
            raise RuntimeError(f"加载测试数据集时出错: {str(e)}")

    def create_test_loader(self, batch_size: int) -> TorchDataLoader:
        """创建测试数据加载器"""
        data_dict = self.load_test_data()
        
        dataset = TimeSeriesDataset(
            data_dict['X_test'][0],
            data_dict['X_test'][1],
            data_dict['X_test'][2],
            data_dict['y_test']
        )
        
        return TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # 测试集不需要打乱顺序
            num_workers=4,
            pin_memory=True
        )