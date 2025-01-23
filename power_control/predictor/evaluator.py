from config import MODEL_CONFIG
from data_processor import DataProcessor, MyDataLoader
from model import NodePredictorNN

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_processor = DataProcessor()
        self.config = MODEL_CONFIG
        self.model = None

    def load_model(self, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None):
        """
        从固定目录加载预训练模型及其优化器和调度器状态
        
        参数:
            optimizer (torch.optim.Optimizer, optional): 优化器实例
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 调度器实例
        """
        # 从配置中获取模型路径
        model_dir = self.config['model_dir']
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录未找到: {model_dir}")
        
        model_path = os.path.join(model_dir, 'checkpoint.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        # 创建新的模型实例
        self.model = NodePredictorNN(
            feature_size=self.data_processor.feature_size
        ).to(self.device)
        
        try:
            # 加载模型和scaler
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功从 {model_path} 加载模型")

            # 加载scaler
            # self.data_processor.feature_scaler = checkpoint['feature_scaler']
            # self.data_processor.dayback_scaler = checkpoint['dayback_scaler']
            # self.data_processor.target_scaler = checkpoint['target_scaler']
            # print("成功加载scaler")

            if optimizer and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("成功加载优化器状态")
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("成功加载调度器状态")
                        
        except Exception as e:
            raise RuntimeError(f"从 {model_path} 加载模型时出错: {str(e)}")
        
        # 设置为评估模式
        self.model.eval()

    def evaluate(self, data_dict: dict) -> tuple:
        """评估模型并生成可视化结果"""
        print("开始评估模型")

        # 1. 加载模型
        self.load_model()

        # 2. 获取测试数据加载器
        test_loader = MyDataLoader().create_one_data_loader(
            data_dict,
            self.config['batch_size'],
            'test'
        )
        
        predictions = []
        targets = []
        
        print("\n=== 模型预测过程调试信息 ===")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # 将数据移到设备上
                past_hour = batch['past_hour'].to(self.device)
                cur_datetime = batch['cur_datetime'].to(self.device)
                dayback = batch['dayback'].to(self.device)
                
                # 获取预测结果
                outputs = self.model(past_hour, cur_datetime, dayback)
                
                # 每隔一定批次打印一些调试信息
                if batch_idx % 10 == 0:
                    print(f"\nBatch {batch_idx}:")
                    print("输出值范围:", outputs.cpu().numpy().min(), "-", outputs.cpu().numpy().max())
                    print("目标值范围:", batch['target'].numpy().min(), "-", batch['target'].numpy().max())
                
                predictions.append(outputs.cpu().numpy())
                targets.append(batch['target'].numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        print("\n=== 数据统计信息 ===")
        print("预测值统计（缩放后）:")
        print(f"预测范围: {predictions.min():.4f} - {predictions.max():.4f}")
        print("\n真实值统计（缩放后）:")
        print(f"真实值范围: {targets.min():.4f} - {targets.max():.4f}")
        
        # 转换回原始尺度并确保非负
        print("\n=== 反标准化过程 ===")
        predictions = self.data_processor.inverse_transform_y(predictions)
        predictions = np.maximum(predictions, 0)  # 确保预测值非负
        targets = self.data_processor.inverse_transform_y(targets)
        
        print("\n预测值统计（原始尺度）:")
        print(f"预测范围: {predictions.min():.4f} - {predictions.max():.4f}")
        print("\n真实值统计（原始尺度）:")
        print(f"真实值范围: {targets.min():.4f} - {targets.max():.4f}")
        
        # 检查唯一值
        print("\n=== 值分布分析 ===")
        print(f"唯一预测值数量: {len(np.unique(predictions))}")
        print(f"唯一真实值数量: {len(np.unique(targets))}")
        
        # 数据有效性检查
        print("\n=== 数据有效性检查 ===")
        print("预测值中的无效值:")
        print(f"NaN: {np.isnan(predictions).sum()}")
        print(f"Inf: {np.isinf(predictions).sum()}")
        print(f"负值: {(predictions < 0).sum()}")
        
        # 计算评估指标
        metrics = {
            'mse': float(mean_squared_error(targets, predictions)),
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
            metrics[key] = float(np.mean(within_tolerance) * 100)
        
        # 添加误差分布统计
        metrics['error_distribution'] = self._calculate_error_distribution(
            targets_discrete, 
            predictions_discrete
        )
        
        # 保存评估结果
        eval_log_dir = os.path.join(self.config['log_dir'], 'eval')
        os.makedirs(eval_log_dir, exist_ok=True)
        
        # 生成可视化结果
        self._plot_error_distribution(predictions - targets, eval_log_dir)
        self._plot_prediction_scatter(targets, predictions, eval_log_dir)
        self._plot_time_series(targets, predictions, eval_log_dir)
        
        print("评估完成。评估结果已保存。")
        
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
    
    def _plot_error_distribution(self, errors, save_dir):
        """绘制误差分布直方图"""
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, edgecolor='black')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'error_distribution.png'))
        plt.close()

    def _plot_prediction_scatter(self, targets, predictions, save_dir):
        """绘制预测散点图"""
        plt.figure(figsize=(10, 6))
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


def main():
    evaluator = Evaluator()
    data_dict = evaluator.data_processor.load_and_prepare_data()
    metrics, predictions, targets = evaluator.evaluate(data_dict)

    print("\n=== 测试结果 ===")
    print("\n基础指标:")
    print(f"均方误差 (MSE): {metrics['mse']:.4f}")
    print(f"均方根误差 (RMSE): {metrics['rmse']:.4f}")
    print(f"平均绝对误差 (MAE): {metrics['mae']:.4f}")
    print(f"R² 分数: {metrics['r2']:.4f}")
    print(f"对称平均绝对百分比误差 (SMAPE): {metrics['smape']:.2f}%")
    
    print("\n准确率指标:")
    print(f"精确匹配: {metrics['accuracy_exact']:.2f}%")
    print(f"误差在 ±1 节点内: {metrics['accuracy_within_1']:.2f}%")
    print(f"误差在 ±2 节点内: {metrics['accuracy_within_2']:.2f}%")
    print(f"误差在 ±5 节点内: {metrics['accuracy_within_5']:.2f}%")
    
    dist = metrics['error_distribution']
    print("\n误差分布:")
    print(f"平均误差: {dist['mean_error']:.2f} 节点")
    print(f"平均绝对误差: {dist['mean_abs_error']:.2f} 节点")
    print(f"中位数误差: {dist['median_error']:.2f} 节点")
    print(f"误差标准差: {dist['error_std']:.2f} 节点")
    print(f"最大误差: {dist['max_error']:.2f} 节点")

if __name__ == '__main__':
    main()