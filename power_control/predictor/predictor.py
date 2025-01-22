from data_processor import DataProcessor
from model import NodePredictorNN
from config import MODEL_CONFIG

import torch
import numpy as np

class NodePredictor:
    def __init__(self):
        """初始化预测器，加载预训练模型和数据处理器"""
        self.config = MODEL_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_processor = DataProcessor()
        
        # 初始化模型
        self.model = NodePredictorNN(
            feature_size=self.data_processor.feature_size
        ).to(self.device)
        
        # 加载预训练模型
        try:
            checkpoint = torch.load(
                f"{self.config['model_dir']}/checkpoint.pth",
                map_location=self.device
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("成功加载预训练模型")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")
        
        # 设置为评估模式
        self.model.eval()
    
    def predict(self, data_path):
        """
        预测未来的计算节点数量
        
        参数:
            data_path (str): CSV数据文件路径
            
        返回:
            float: 预测的计算节点数量
        """
        try:
            # 1. 从CSV文件加载并处理数据
            past_hour_data, cur_datetime, dayback_data = self.data_processor.process_single_record(data_path)
            
            # 2. 转换为张量
            past_hour_tensor = torch.FloatTensor(past_hour_data).unsqueeze(0).to(self.device)
            cur_datetime_tensor = torch.FloatTensor(cur_datetime).unsqueeze(0).to(self.device)
            dayback_tensor = torch.FloatTensor(dayback_data).unsqueeze(0).to(self.device)
            
            # 3. 模型预测
            with torch.no_grad():
                prediction_scaled = self.model(
                    past_hour_tensor,
                    cur_datetime_tensor,
                    dayback_tensor
                )
            
            # 4. 反向转换预测结果
            prediction = self.data_processor.inverse_transform_y(
                prediction_scaled.cpu().numpy()
            )
            
            # 5. 确保预测结果为非负整数
            final_prediction = max(0, round(float(prediction[0][0])))
            
            return final_prediction
            
        except Exception as e:
            print(f"预测过程出错: {str(e)}")
            # 发生错误时返回一个安全的默认值
            return 0
    
    def predict_batch(self, data_paths):
        """
        批量预测未来的计算节点数量
        
        参数:
            data_paths (list): CSV数据文件路径列表
            
        返回:
            np.ndarray: 预测的计算节点数量数组
        """
        try:
            # 1. 处理所有数据文件
            batch_data = [self.data_processor.process_single_record(path) for path in data_paths]
            past_hour_batch = np.stack([data[0] for data in batch_data])
            cur_datetime_batch = np.stack([data[1] for data in batch_data])
            dayback_batch = np.stack([data[2] for data in batch_data])
            
            # 2. 转换为张量
            past_hour_tensor = torch.FloatTensor(past_hour_batch).to(self.device)
            cur_datetime_tensor = torch.FloatTensor(cur_datetime_batch).to(self.device)
            dayback_tensor = torch.FloatTensor(dayback_batch).to(self.device)
            
            # 3. 模型预测
            with torch.no_grad():
                predictions_scaled = self.model(
                    past_hour_tensor,
                    cur_datetime_tensor,
                    dayback_tensor
                )
            
            # 4. 反向转换预测结果
            predictions = self.data_processor.inverse_transform_y(
                predictions_scaled.cpu().numpy()
            )
            
            # 5. 确保预测结果为非负整数
            final_predictions = np.maximum(0, np.round(predictions)).astype(int)
            
            return final_predictions.flatten()
            
        except Exception as e:
            print(f"批量预测过程出错: {str(e)}")
            # 发生错误时返回一个安全的默认值数组
            return np.zeros(len(data_paths))
