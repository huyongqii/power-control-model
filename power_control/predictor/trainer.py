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
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, y_pred, y_true):
        # 检查并处理无效值
        if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
            print("警告：预测值中存在 NaN 或 Inf")
            y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 基础损失：使用 Huber Loss 代替 MSE，对异常值更不敏感
        base_loss = F.huber_loss(y_pred, y_true, delta=1.0)
        
        # 离散化损失：添加数值稳定性处理
        discretization_loss = torch.mean(
            torch.clamp(torch.abs(y_pred - torch.round(y_pred)), max=10.0)
        )
        
        # 极端误差惩罚：添加上下限
        error = torch.clamp(torch.abs(y_pred - y_true), max=50.0)
        extreme_error_mask = error > 10
        extreme_error_loss = torch.mean(
            torch.where(extreme_error_mask, error, torch.zeros_like(error))
        )
        
        # 非负约束：添加上限
        non_negative_loss = torch.mean(
            torch.clamp(F.relu(-y_pred), max=10.0)
        )
        
        # 使用较小的权重系数
        total_loss = base_loss + \
                    0.01 * discretization_loss + \
                    0.05 * extreme_error_loss + \
                    0.1 * non_negative_loss
        
        # 检查损失值
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("警告：损失值为 NaN 或 Inf")
            print(f"Base Loss: {base_loss.item()}")
            print(f"Discretization Loss: {discretization_loss.item()}")
            print(f"Extreme Error Loss: {extreme_error_loss.item()}")
            print(f"Non-negative Loss: {non_negative_loss.item()}")
            
            # 如果发生 NaN，返回一个可以继续训练的值
            total_loss = torch.tensor(1.0, device=total_loss.device, requires_grad=True)
        
        return total_loss, {
            'base_loss': base_loss.item(),
            'discretization_loss': discretization_loss.item(),
            'extreme_error_loss': extreme_error_loss.item(),
            'non_negative_loss': non_negative_loss.item(),
            'total_loss': total_loss.item()
        }

class Trainer:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_processor = DataProcessor()
        self.model = None
        self.criterion = CustomEnergyLoss()
        # 更新损失历史记录的组件
        self.loss_history = {
            'base_loss': [],
            'discretization_loss': [],
            'extreme_error_loss': [],
            'non_negative_loss': [],
            'total_loss': []
        }

    def train(self, data_dict: dict):
        """训练模型并记录训练过程"""
        print(f"开始在设备 {self.device} 上训练模型")
        
        # 初始化模型
        self.model = NodePredictorNN(
            feature_size=self.data_processor.feature_size
        ).to(self.device)
        
        # 创建数据加载器
        data_loaders = MyDataLoader().create_data_loaders(
            data_dict,
            self.config['batch_size']
        )
        
        # 优化器和学习率调度器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config['lr_factor'],
            patience=self.config['lr_patience'],
            min_lr=self.config['min_lr']
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            # 训练阶段
            self.model.train()
            train_loss = 0
            batch_count = len(data_loaders['train'])
            
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
                    if key in self.loss_history:  # 确保键存在
                        self.loss_history[key].append(value)
                
                loss.backward()
                
                # 使用配置中的梯度裁剪参数
                if self.config['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                
                optimizer.step()
                
                train_loss += loss_components['total_loss']
                
                progress = int(50 * batch_idx / batch_count)
                print(f"\r进度: [{'=' * progress}{' ' * (50-progress)}] {batch_idx}/{batch_count} "
                      f"- 当前 loss: {loss_components['total_loss']:.6f}", end="")
            
            train_loss /= len(data_loaders['train'])
            
            # 验证阶段
            val_loss = self._validate(data_loaders['val'], self.criterion)
            
            # 在验证后调整学习率
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            history = {
                'train_loss': [train_loss],
                'val_loss': [val_loss],
                'learning_rates': [current_lr]
            }
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0

                self.save_model(optimizer, scheduler)
                print(f"发现更好的模型！已保存模型")
            else:
                patience_counter += 1
            
            # 打印详细的训练信息
            print(f"\nEpoch {epoch}")
            print(f"Train Loss: {train_loss/batch_count:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Best Val Loss: {best_val_loss:.6f}")
            
            # 早停检查
            if patience_counter >= self.config['early_stopping_patience']:
                print("Early stopping triggered")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
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
                loss, loss_components = criterion(outputs, targets)
            
            total_loss += loss_components['total_loss']
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
        for key in ['base_loss', 'discretization_loss', 'extreme_error_loss', 'non_negative_loss']:  # 添加新的损失组件
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
