import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from data_processor import DataProcessor

# 设置matplotlib的后端，避免在没有显示设备的环境中出错
plt.switch_backend('agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR,'power_control', 'predictor', 'data')
MODEL_DIR = os.path.join(BASE_DIR,'power_control', 'predictor', 'models')
LOG_DIR = os.path.join(BASE_DIR,'power_control', 'predictor', 'logs')

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
    'data_path': os.path.join(DATA_DIR, 'training_data_20250115_195652.csv'),
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
        
    def train(self, data_dict: dict):
        """训练模型，使用交叉验证来选择最佳模型"""
        print(f"Training on device: {self.device}")
        
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        X_val = data_dict['X_val']
        y_val = data_dict['y_val']
        
        # 转换为PyTorch张量
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        epochs = self.config['epochs']
        batch_size = self.config['batch_size']
        
        # 创建模型
        self.model = NodePredictorNN(X_train.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练一个epoch
            train_loss = self.train_one_epoch(
                torch.FloatTensor(X_train).to(self.device),
                torch.FloatTensor(y_train).to(self.device),
                criterion, optimizer, batch_size
            )
            
            # 在验证集上评估
            val_loss = self.validate(X_val_tensor, y_val_tensor, criterion)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 使用最佳模型状态
        self.model.load_state_dict(best_model_state)
        torch.save(best_model_state, self.config['model_path'])
        print(f"\nBest validation loss: {best_val_loss:.4f}")

    def evaluate(self, data_dict: dict) -> tuple:
        """评估模型性能"""
        self.model.eval()
        X_test = torch.FloatTensor(data_dict['X_test']).to(self.device)
        y_test = data_dict['y_test']
        
        with torch.no_grad():
            predictions = self.model(X_test).cpu().numpy()
        
        # 转换回原始尺度
        predictions = self.data_processor.inverse_transform_y(predictions)
        y_test = self.data_processor.inverse_transform_y(y_test)
        
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
        
        # 计算准确率
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

    def evaluate_with_cv(self, data_dict: dict) -> dict:
        """使用交叉验证进行模型评估"""
        X_train = data_dict['X_train']
        y_train = data_dict['y_train']
        n_splits = self.config.get('n_splits', 5)
        
        # 存储所有折的评估结果
        cv_metrics = {
            'mse': [], 'mae': [], 'r2': [], 'accuracy': [],
            'within_1_node': [], 'within_2_nodes': [], 'within_5_nodes': []
        }
        
        # 交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            # 获取当前折的数据
            X_val_fold = X_train[val_idx]
            y_val_fold = y_train[val_idx]
            
            # 在当前折上评估
            X_val_tensor = torch.FloatTensor(X_val_fold).to(self.device)
            with torch.no_grad():
                predictions = self.model(X_val_tensor).cpu().numpy()
            
            # 转换回原始尺度
            predictions = self.data_processor.inverse_transform_y(predictions)
            y_val_true = self.data_processor.inverse_transform_y(y_val_fold)
            
            # 确保预测值为非负整数
            predictions = np.maximum(0, np.round(predictions)).astype(int).reshape(-1)
            y_val_true = y_val_true.reshape(-1)
            
            # 计算当前折的指标
            mse = mean_squared_error(y_val_true, predictions)
            mae = mean_absolute_error(y_val_true, predictions)
            r2 = r2_score(y_val_true, predictions)
            
            # 计算准确率
            abs_errors = np.abs(predictions - y_val_true)
            cv_metrics['mse'].append(mse)
            cv_metrics['mae'].append(mae)
            cv_metrics['r2'].append(r2)
            cv_metrics['within_1_node'].append(np.mean(abs_errors <= 1) * 100)
            cv_metrics['within_2_nodes'].append(np.mean(abs_errors <= 2) * 100)
            cv_metrics['within_5_nodes'].append(np.mean(abs_errors <= 5) * 100)
        
        # 计算平均指标和标准差
        final_metrics = {}
        for metric in cv_metrics:
            values = cv_metrics[metric]
            final_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return final_metrics

    def train_one_epoch(self, X_train, y_train, criterion, optimizer, batch_size):
        """训练一个epoch"""
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
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
        return total_loss / batch_count if batch_count > 0 else float('inf')
    
    def validate(self, X_val, y_val, criterion):
        """验证模型性能"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            val_loss = criterion(outputs, y_val)
        return val_loss.item()

def main():
    """测试模型训练和评估"""
    predictor = NodePredictor(MODEL_CONFIG)
    
    # 加载和准备数据
    data_dict = predictor.data_processor.load_and_prepare_data(MODEL_CONFIG['data_path'])
    
    # 打印数据形状
    print("\nData shapes:")
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
    
    # 训练模型
    predictor.train(data_dict)
    
    # 在测试集上进行最终评估
    test_metrics, predictions, y_test = predictor.evaluate(data_dict)
    
    # 打印测试集结果
    print("\n=== Test Set Results ===")
    print(f"Mean Squared Error: {test_metrics['mse']:.2f}")
    print(f"Mean Absolute Error: {test_metrics['mae']:.2f}")
    print(f"R² Score: {test_metrics['r2']:.2f}")
    print(f"Accuracy (±{MODEL_CONFIG.get('accuracy_tolerance', 2)} nodes): {test_metrics['accuracy']:.2f}%")
    
    dist = test_metrics['error_distribution']
    print("\nError Distribution:")
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