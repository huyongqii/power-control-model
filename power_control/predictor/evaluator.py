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
            # 使用 weights_only=True 来安全加载模型
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功从 {model_path} 加载模型")

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
        """评估模型并生成可视化结果，分别评估最小值和最大值的预测效果"""
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
        
        with torch.no_grad():
            for batch in test_loader:
                # 将数据移到设备上
                past_hour = batch['past_hour'].to(self.device)
                cur_datetime = batch['cur_datetime'].to(self.device)
                dayback = batch['dayback'].to(self.device)
                
                # 获取预测结果
                outputs = self.model(past_hour, cur_datetime, dayback)
                predictions.append(outputs.cpu().numpy())
                targets.append(batch['target'].numpy())
        
        # 合并批次结果
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # 转换回原始尺度
        predictions = self.data_processor.inverse_transform_y(predictions)
        targets = self.data_processor.inverse_transform_y(targets)
        
        # 分别计算最小值和最大值的评估指标
        metrics = {
            'min_value': {
                'mse': float(mean_squared_error(targets[:, 0], predictions[:, 0])),
                'mae': float(mean_absolute_error(targets[:, 0], predictions[:, 0])),
                'rmse': float(np.sqrt(mean_squared_error(targets[:, 0], predictions[:, 0]))),
                'r2': float(r2_score(targets[:, 0], predictions[:, 0])),
                'smape': float(self._calculate_smape(targets[:, 0], predictions[:, 0]))
            },
            'max_value': {
                'mse': float(mean_squared_error(targets[:, 1], predictions[:, 1])),
                'mae': float(mean_absolute_error(targets[:, 1], predictions[:, 1])),
                'rmse': float(np.sqrt(mean_squared_error(targets[:, 1], predictions[:, 1]))),
                'r2': float(r2_score(targets[:, 1], predictions[:, 1])),
                'smape': float(self._calculate_smape(targets[:, 1], predictions[:, 1]))
            }
        }
        
        # 计算不同容差下的准确率
        predictions_discrete = np.maximum(0, np.round(predictions))
        targets_discrete = np.maximum(0, np.round(targets))
        
        for value_type, col_idx in [('min_value', 0), ('max_value', 1)]:
            for tolerance in [0, 1, 2, 5]:
                if tolerance == 0:
                    key = 'accuracy_exact'
                else:
                    key = f'accuracy_within_{tolerance}'
                within_tolerance = np.abs(predictions_discrete[:, col_idx] - targets_discrete[:, col_idx]) <= tolerance
                metrics[value_type][key] = float(np.mean(within_tolerance) * 100)
        
            # 添加误差分布统计
            metrics[value_type]['error_distribution'] = self._calculate_error_distribution(
                targets_discrete[:, col_idx], 
                predictions_discrete[:, col_idx]
            )
        
        # 保存评估指标
        eval_log_dir = os.path.join(self.config['log_dir'], 'eval')
        if not os.path.exists(eval_log_dir):
            os.makedirs(eval_log_dir)
        
        # 生成可视化结果
        self._plot_error_distributions(predictions - targets, eval_log_dir)
        self._plot_prediction_scatters(targets, predictions, eval_log_dir)
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
    
    def _plot_error_distributions(self, errors, save_dir):
        """分别绘制最小值和最大值的误差分布直方图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.hist(errors[:, 0], bins=50, edgecolor='black')
        ax1.set_title('Min Value Error Distribution')
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Frequency')
        ax1.grid(True)
        
        ax2.hist(errors[:, 1], bins=50, edgecolor='black')
        ax2.set_title('Max Value Error Distribution')
        ax2.set_xlabel('Error')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'error_distributions.png'))
        plt.close()

    def _plot_prediction_scatters(self, targets, predictions, save_dir):
        """分别绘制最小值和最大值的预测散点图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        ax1.scatter(targets[:, 0], predictions[:, 0], alpha=0.5)
        ax1.plot([min(targets[:, 0]), max(targets[:, 0])], 
                 [min(targets[:, 0]), max(targets[:, 0])], 'r--')
        ax1.set_title('Min Value: Predicted vs Actual')
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.grid(True)
        
        ax2.scatter(targets[:, 1], predictions[:, 1], alpha=0.5)
        ax2.plot([min(targets[:, 1]), max(targets[:, 1])], 
                 [min(targets[:, 1]), max(targets[:, 1])], 'r--')
        ax2.set_title('Max Value: Predicted vs Actual')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'prediction_scatters.png'))
        plt.close()

    def _plot_time_series(self, targets, predictions, save_dir):
        """分别绘制最小值和最大值的时间序列对比图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        ax1.plot(targets[:, 0], label='Actual Min', alpha=0.7)
        ax1.plot(predictions[:, 0], label='Predicted Min', alpha=0.7)
        ax1.set_title('Time Series Prediction - Min Values')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Number of Nodes')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(targets[:, 1], label='Actual Max', alpha=0.7)
        ax2.plot(predictions[:, 1], label='Predicted Max', alpha=0.7)
        ax2.set_title('Time Series Prediction - Max Values')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Number of Nodes')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'time_series.png'))
        plt.close()


def main():
    evaluator = Evaluator()
    data_dict = evaluator.data_processor.load_and_prepare_data()
    test_metrics, predictions, targets = evaluator.evaluate(data_dict)

    for value_type in ['min_value', 'max_value']:
        type_name = "最小值" if value_type == 'min_value' else "最大值"
        print(f"\n=== {type_name}测试结果 ===")
        
        print("\n基础指标:")
        print(f"均方误差 (MSE): {test_metrics[value_type]['mse']:.4f}")
        print(f"均方根误差 (RMSE): {test_metrics[value_type]['rmse']:.4f}")
        print(f"平均绝对误差 (MAE): {test_metrics[value_type]['mae']:.4f}")
        print(f"R² 分数: {test_metrics[value_type]['r2']:.4f}")
        print(f"对称平均绝对百分比误差 (SMAPE): {test_metrics[value_type]['smape']:.2f}%")
        
        print("\n准确率指标:")
        print(f"精确匹配: {test_metrics[value_type]['accuracy_exact']:.2f}%")
        print(f"误差在 ±1 节点内: {test_metrics[value_type]['accuracy_within_1']:.2f}%")
        print(f"误差在 ±2 节点内: {test_metrics[value_type]['accuracy_within_2']:.2f}%")
        print(f"误差在 ±5 节点内: {test_metrics[value_type]['accuracy_within_5']:.2f}%")
        
        # 打印误差分布
        dist = test_metrics[value_type]['error_distribution']
        print("\n误差分布:")
        print(f"平均误差: {dist['mean_error']:.2f} 节点")
        print(f"平均绝对误差: {dist['mean_abs_error']:.2f} 节点")
        print(f"中位数误差: {dist['median_error']:.2f} 节点")
        print(f"误差标准差: {dist['error_std']:.2f} 节点")
        print(f"最大误差: {dist['max_error']:.2f} 节点")
        
        print("\n误差百分位数:")
        print(f"25分位数: {dist['error_25th']:.2f} 节点")
        print(f"50分位数: {dist['error_50th']:.2f} 节点")
        print(f"75分位数: {dist['error_75th']:.2f} 节点")
        print(f"90分位数: {dist['error_90th']:.2f} 节点")
        print(f"95分位数: {dist['error_95th']:.2f} 节点")
        
        print("\n预测偏差:")
        print(f"精确预测比例: {dist['exact_prediction']:.2f}%")
        print(f"过预测比例: {dist['over_prediction']:.2f}%")
        print(f"欠预测比例: {dist['under_prediction']:.2f}%")
        
        print("\n误差范围分布:")
        print(f"0-1 节点: {dist['error_0_to_1']:.2f}%")
        print(f"1-2 节点: {dist['error_1_to_2']:.2f}%")
        print(f"2-5 节点: {dist['error_2_to_5']:.2f}%")
        print(f"5-10 节点: {dist['error_5_to_10']:.2f}%")
        print(f">10 节点: {dist['error_10_to_inf']:.2f}%")
        
        print("\n" + "="*50)  # 分隔线

if __name__ == '__main__':
    main()