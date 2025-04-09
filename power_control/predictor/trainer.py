import os
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from model import NodePredictorNN
from config import MODEL_CONFIG
from data_loader import DataLoader


plt.switch_backend('agg')

class CustomEnergyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y_pred, y_true):
        # 确保形状匹配：将 y_pred 从 [batch_size, 1] 转换为 [batch_size]
        # y_pred = y_pred.squeeze()
        
        # 使用 Huber Loss 作为主要损失函数
        base_loss = F.huber_loss(y_pred, y_true, delta=1.0)
        
        # 添加轻微的平滑项
        smoothness_loss = torch.mean(torch.abs(y_pred - torch.round(y_pred)))
        
        # 总损失，降低平滑项的权重
        total_loss = base_loss + 0.001 * smoothness_loss
        
        return total_loss, {
            'base_loss': base_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'total_loss': total_loss.item()
        }

class Trainer:
    def __init__(self):
        self.config = MODEL_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_loader = DataLoader()
        self.model = None
        self.criterion = CustomEnergyLoss()
        # 添加训练和验证损失的记录
        self.loss_history = {
            'train_loss': [],
            'val_loss': []
        }

    def train(self):
        """训练模型并记录训练过程"""
        print(f"开始在设备 {self.device} 上训练模型")
        
        # 初始化模型
        self.model = NodePredictorNN(
            feature_size=self.data_loader.feature_size
        ).to(self.device)
        
        # 创建训练集和验证集的数据加载器
        train_loader = self.data_loader.create_data_loaders(
            batch_size=self.config['batch_size'],
            split='train'
        )['train']
        
        val_loader = self.data_loader.create_data_loaders(
            batch_size=self.config['batch_size'],
            split='val'
        )['val']
        
        data_loaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # 使用 AdamW 优化器
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 学习率预热
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config['warmup_epochs']
        )
        
        # 主学习率调度器
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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
                
                # # 记录损失组件
                # for key, value in loss_components.items():
                #     if key in self.loss_history:  # 确保键存在
                #         self.loss_history[key].append(value)
                
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
            
            # 验证阶段
            val_loss = self._validate(data_loaders['val'], self.criterion)
            
            # 更新学习率
            if epoch <= self.config['warmup_epochs']:
                warmup_scheduler.step()
            else:
                main_scheduler.step(val_loss)
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

                self.save_model(optimizer, main_scheduler, epoch, best_val_loss)
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
            
            # 在每个epoch结束后记录损失
            epoch_train_loss = train_loss / batch_count
            epoch_val_loss = val_loss
            
            self.loss_history['train_loss'].append(epoch_train_loss)
            self.loss_history['val_loss'].append(epoch_val_loss)
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 绘制训练历史
        train_log_dir = self.config['log_dir']
        if not os.path.exists(train_log_dir):
            os.makedirs(train_log_dir)
        # self._plot_training_history(history, train_log_dir)
        
        # 保存训练历史
        history_path = os.path.join(train_log_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        print("训练完成。训练历史已保存。")
        
        # 在训练结束后绘制损失曲线
        self._plot_loss_curves()
        
        return history

    def save_model(self, optimizer, scheduler, epoch, best_val_loss):
        """保存模型检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'loss_history': self.loss_history
        }
        
        save_path = os.path.join(self.config['model_dir'], 'checkpoint.pth')
        torch.save(checkpoint, save_path)
        print(f"模型检查点已保存到: {save_path}")

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
        for key in ['base_loss', 'smoothness_loss']:  # 添加新的损失组件
            ax2.plot(self.loss_history[key], label=key)
        ax2.set_title('Loss Components During Training')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'detailed_training_history.png'))
        plt.close()

    def _plot_loss_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history['train_loss'], label='Training Loss')
        plt.plot(self.loss_history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config['model_dir'], 'loss_curves.png'))
        plt.close()
        print(f"Loss curves saved to: {self.config['model_dir']}/loss_curves.png")

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
    history = predictor.train()

if __name__ == '__main__':
    main()
