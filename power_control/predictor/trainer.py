from model import NodePredictorNN
from config import MODEL_CONFIG
from data_processor import DataProcessor, MyDataLoader

import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')

class CustomEnergyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, delta=2.0):
        """
        能源预测专用的损失函数
        
        参数:
            alpha (float): 过预测惩罚权重
            beta (float): 欠预测惩罚权重
            gamma (float): 平滑度惩罚权重
            delta (float): 负值惩罚权重
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
    
    def forward(self, y_pred, y_true):
        """
        计算损失值
        
        参数:
            y_pred (torch.Tensor): 预测值 [batch_size, sequence_length]
            y_true (torch.Tensor): 真实值 [batch_size, sequence_length]
        """
        # 基础MSE损失
        base_loss = F.mse_loss(y_pred, y_true, reduction='none')
        
        # 过预测惩罚（预测值大于真实值）
        over_prediction = torch.max(y_pred - y_true, torch.zeros_like(y_pred))
        over_prediction_loss = torch.mean(over_prediction ** 2)
        
        # 欠预测惩罚（预测值小于真实值）
        under_prediction = torch.max(y_true - y_pred, torch.zeros_like(y_pred))
        under_prediction_loss = torch.mean(under_prediction ** 2)
        
        # 负值惩罚
        negative_values = torch.max(-y_pred, torch.zeros_like(y_pred))
        negative_penalty = torch.mean(negative_values ** 2)
        
        # 平滑度惩罚（相邻预测值的差异）
        # 确保在时间维度上计算差分
        if len(y_pred.shape) == 3:  # [batch_size, sequence_length, features]
            smoothness_loss = torch.mean(torch.diff(y_pred, dim=1) ** 2)
        else:  # [batch_size, sequence_length]
            smoothness_loss = torch.mean(torch.diff(y_pred, dim=1) ** 2)
        
        # 总损失
        total_loss = (torch.mean(base_loss) + 
                     self.alpha * over_prediction_loss +
                     self.beta * under_prediction_loss +
                     self.gamma * smoothness_loss +
                     self.delta * negative_penalty)
        
        # 记录各个损失组件（用于监控）
        loss_components = {
            'base_loss': torch.mean(base_loss).item(),
            'over_prediction': over_prediction_loss.item(),
            'under_prediction': under_prediction_loss.item(),
            'smoothness': smoothness_loss.item(),
            'negative_penalty': negative_penalty.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components

class Trainer:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_processor = DataProcessor()
        self.model = None
        # 调整权重以更强烈地惩罚负值和过预测
        self.criterion = CustomEnergyLoss(
            alpha=1.2,    # 过预测惩罚
            beta=0.8,     # 欠预测惩罚
            gamma=0.3,    # 平滑度惩罚
            delta=2.0     # 负值惩罚
        )
        self.loss_history = {
            'base_loss': [],
            'over_prediction': [],
            'under_prediction': [],
            'smoothness': [],
            'negative_penalty': [],
            'total_loss': []
        }

    def train(self, data_dict: dict):
        """训练模型并记录训练过程"""
        print(f"开始在设备 {self.device} 上训练模型")
        
        data_loaders = MyDataLoader().create_data_loaders(
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
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
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
                
                past_hour = batch['past_hour'].to(self.device)
                cur_datetime = batch['cur_datetime'].to(self.device)
                dayback = batch['dayback'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(past_hour, cur_datetime, dayback)
                loss, loss_components = self.criterion(outputs, targets)
                
                # 记录损失组件
                for key, value in loss_components.items():
                    self.loss_history[key].append(value)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss_components['total_loss']
                
                progress = int(50 * batch_idx / batch_count)
                print(f"\r进度: [{'=' * progress}{' ' * (50-progress)}] {batch_idx}/{batch_count} "
                    f"- 当前 loss: {loss_components['total_loss']:.6f}", end="")
            
            train_loss /= len(data_loaders['train'])
            
            # 验证阶段
            val_loss = self._validate(data_loaders['val'], self.criterion)
            
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
                print(f"Epoch {epoch}: Val Loss improved to {val_loss:.6f}. 模型已保存。")
            else:
                patience_counter += 1
                print(f"Epoch {epoch}: Val Loss = {val_loss:.6f} 不改善。")
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"早停触发于 epoch {epoch}")
                    break
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # 绘制训练历史
        train_log_dir = self.config['log_dir']
        if not os.path.exists(train_log_dir):
            os.makedirs(train_log_dir)
        self._plot_training_history(history, train_log_dir)
        
        # 保存训练历史
        history_path = os.path.join(train_log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        print("训练完成。训练历史已保存。")
        
        return history

    def save_model(self, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler._LRScheduler = None):
        """
        保存模型、优化器和调度器的状态及配置到固定目录
        
        参数:
            optimizer (torch.optim.Optimizer, optional): 优化器实例
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 调度器实例
        """
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        # 使用配置中的模型目录
        model_dir = self.config['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'checkpoint.pth')
        config_path = os.path.join(model_dir, 'config.json')
        
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
            
            # 保存检查点和配置
            torch.save(checkpoint, model_path)
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            print(f"模型、优化器和调度器状态已保存到 {model_dir}")
            
        except Exception as e:
            raise RuntimeError(f"保存模型时出错: {str(e)}")

    def _validate(self, data_loader, criterion):
        """验证模型性能"""
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        for batch in data_loader:
            # 将数据移到设备上
            past_hour = batch['past_hour'].to(self.device)
            cur_datetime = batch['cur_datetime'].to(self.device)
            dayback = batch['dayback'].to(self.device)
            targets = batch['target'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(past_hour, cur_datetime, dayback)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else float('inf')

    def _plot_training_history(self, history, save_path):
        """绘制更详细的训练历史"""
        plt.figure(figsize=(15, 10))
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # 绘制总损失
        ax1.plot(self.loss_history['total_loss'], label='Total Loss')
        ax1.set_title('Total Loss During Training')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制损失组件
        for key in ['base_loss', 'over_prediction', 'under_prediction', 
                   'smoothness', 'negative_penalty']:
            ax2.plot(self.loss_history[key], label=key)
        ax2.set_title('Loss Components During Training')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'detailed_training_history.png'))
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
    set_seed(42)

    predictor = Trainer()
    data_dict = predictor.data_processor.load_and_prepare_data()
    history = predictor.train(data_dict)

if __name__ == '__main__':
    main()
