import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
import matplotlib.pyplot as plt

# 添加父目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processor import DataProcessor

# 设置matplotlib的后端，避免在没有显示设备的环境中出错
plt.switch_backend('agg')

# 全局配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR,'power_control','data', 'processed_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 模型配置
MODEL_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    'model_path': os.path.join(MODEL_DIR, 'node_predictor.pth'),
    'data_path': os.path.join(DATA_DIR, 'training_data_20250115_192555.csv'),
    'log_path': os.path.join(LOG_DIR, 'training.log')
}

class NodePredictorNN(nn.Module):
    """深度学习节点预测模型"""
    
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            # 第一层
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第二层
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 输出层
            nn.Linear(32, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
            
    def forward(self, x):
        return self.model(x)

class NodePredictor:
    """计算节点数预测器"""
    
    def __init__(self, config: dict = None):
        self.config = config or MODEL_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        # 初始化数据处理器
        self.data_processor = DataProcessor(self.config)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练预测模型"""
        print(f"Training on device: {self.device}")
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        
        # 创建模型
        self.model = NodePredictorNN(X_train.shape[1]).to(self.device)
        
        # 使用MSE损失
        criterion = nn.MSELoss()
        
        # 使用Adam优化器，较小的学习率
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        epochs = self.config['epochs']
        batch_size = self.config['batch_size']
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0
            
            # 打乱数据
            indices = torch.randperm(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 检查损失值是否有效
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                else:
                    print(f"Warning: Invalid loss value detected: {loss.item()}")
                    continue
            
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                
                # 早停
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), self.config['model_path'])
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['early_stopping_patience']:
                        print("Early stopping triggered")
                        break
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> tuple:
        """评估模型性能"""
        self.model.eval()
        X_test = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_test).cpu().numpy()
        
        # 将预测值和真实值转换回原始尺度
        predictions = self.data_processor.inverse_transform_y(predictions)
        print(predictions)
        y_test = self.data_processor.inverse_transform_y(y_test)
        print(y_test)
        
        # 确保预测值为非负整数
        predictions = np.maximum(0, np.round(predictions)).astype(int).reshape(-1)
        y_test = y_test.reshape(-1)
        
        # 计算基础指标
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # 计算误差分布
        errors = predictions - y_test
        abs_errors = np.abs(errors)
        
        error_distribution = {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'max_error': np.max(abs_errors),
            'error_std': np.std(errors),
            'within_1_node': np.mean(abs_errors <= 1) * 100,
            'within_2_nodes': np.mean(abs_errors <= 2) * 100,
            'within_5_nodes': np.mean(abs_errors <= 5) * 100
        }
        
        # 计算准确率（在容差范围内的预测比例）
        tolerance = self.config.get('accuracy_tolerance', 2)
        accuracy = np.mean(abs_errors <= tolerance) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'error_distribution': error_distribution
        }
        
        return metrics, predictions, y_test
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测计算节点数"""
        self.model.eval()
        X = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
        
        return np.maximum(0, np.round(predictions)).astype(int).reshape(-1)
        
    def save_model(self, path: str = None):
        """保存模型"""
        path = path or self.config['model_path']
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str = None):
        """加载模型"""
        path = path or self.config['model_path']
        self.model.load_state_dict(torch.load(path))

def main():
    """测试模型训练和评估"""
    # 加载和准备数据
    predictor = NodePredictor(MODEL_CONFIG)
    X_train, X_test, y_train, y_test = predictor.data_processor.load_and_prepare_data(
        MODEL_CONFIG['data_path']
    )
    
    # 打印数据形状
    print("\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # 训练模型
    predictor.train(X_train, y_train)
    
    # 评估模型
    metrics, predictions, y_test = predictor.evaluate(X_test, y_test)
    
    # 打印详细评估结果
    print("\n=== Model Evaluation Results ===")
    print(f"\nBasic Metrics:")
    print(f"Mean Squared Error: {metrics['mse']:.2f}")
    print(f"Mean Absolute Error: {metrics['mae']:.2f}")
    print(f"R² Score: {metrics['r2']:.2f}")
    print(f"Accuracy (±{MODEL_CONFIG.get('accuracy_tolerance', 2)} nodes): {metrics['accuracy']:.2f}%")
    
    print("\nError Distribution:")
    dist = metrics['error_distribution']
    print(f"Mean Error: {dist['mean_error']:.2f} nodes")
    print(f"Median Error: {dist['median_error']:.2f} nodes")
    print(f"Max Error: {dist['max_error']:.2f} nodes")
    print(f"Error Std: {dist['error_std']:.2f} nodes")
    print(f"Predictions within ±1 node: {dist['within_1_node']:.2f}%")
    print(f"Predictions within ±2 nodes: {dist['within_2_nodes']:.2f}%")
    print(f"Predictions within ±5 nodes: {dist['within_5_nodes']:.2f}%")
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(y_test[:100], label='Actual', marker='o')
    plt.plot(predictions[:100], label='Predicted', marker='x')
    plt.title('Actual vs Predicted Computing Nodes (First 100 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Number of Computing Nodes')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(LOG_DIR, 'prediction_visualization.png'))
    plt.close()
    
    # 绘制误差分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(np.abs(predictions - y_test), bins=30, edgecolor='black')
    plt.title('Prediction Error Distribution')
    plt.xlabel('Absolute Error (nodes)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(LOG_DIR, 'error_distribution.png'))
    plt.close()

if __name__ == "__main__":
    main() 