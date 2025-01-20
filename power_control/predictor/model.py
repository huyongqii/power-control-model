import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        assert channels % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        B, C, L = x.size()
        
        # 多头注意力计算
        q = self.query(x).view(B, self.num_heads, self.head_dim, L).permute(0, 1, 3, 2)  # B, H, L, D
        k = self.key(x).view(B, self.num_heads, self.head_dim, L)  # B, H, D, L
        v = self.value(x).view(B, self.num_heads, self.head_dim, L)  # B, H, D, L
        
        # 计算注意力权重
        energy = torch.matmul(q, k) / math.sqrt(self.head_dim)  # B, H, L, L
        attention = self.softmax(energy)  # B, H, L, L
        
        # 应用注意力
        out = torch.matmul(attention, v.permute(0, 1, 3, 2))  # B, H, L, D
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, L)  # B, C, L
        
        # 残差连接
        out = self.gamma * out + x
        
        return out

class NodePredictorNN(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        
        # 1. 历史序列编码器
        self.past_encoder = nn.Sequential(
            # 第一层卷积块
            nn.Conv1d(feature_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),  # 恢复使用 BatchNorm
            nn.Dropout(0.2),     # 保持较小的 dropout
            
            # 第二层卷积块
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            # 第三层卷积块
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            # 多头自注意力，但减少头数
            MultiHeadSelfAttention(256, num_heads=4),  # 减少到4个头
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 2. 时间特征编码器
        self.datetime_encoder = nn.Sequential(
            nn.Linear(3, 64), # 处理时、日、周特征
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # 3. 历史模式编码器
        self.dayback_encoder = nn.Sequential(
            nn.Linear(8, 64), # 处理历史同时段特征
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        # 简化融合层
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        
        # 简化预测头
        self.predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.ReLU()  # 使用 ReLU 确保非负输出
        )
        
    def forward(self, past_hour, cur_datetime, dayback):
        # 处理历史序列
        past_hour = past_hour.transpose(1, 2)  # [B, C, L]
        x = past_hour
        
        # 应用三个卷积块和注意力
        x = self.past_encoder[0:4](x)  # 第一个卷积块 [B, 64, L]
        x = self.past_encoder[4:8](x)  # 第二个卷积块 [B, 128, L]
        x = self.past_encoder[8:11](x)  # 第三个卷积块 [B, 256, L]
        x = self.past_encoder[11:](x)   # 注意力和池化 [B, 256, 1]
        past_features = x.squeeze(-1)    # [B, 256]
        
        # 处理时间特征和历史模式特征
        datetime_features = self.datetime_encoder(cur_datetime)  # [B, 128]
        dayback_features = self.dayback_encoder(dayback)  # [B, 128]
        
        # 特征融合
        combined = torch.cat([past_features, datetime_features, dayback_features], dim=1)  # [B, 512]
        fused = self.fusion(combined)
        
        # 预测
        pred = self.predictor(fused)
        pred = torch.clamp(pred, min=0.0, max=1000.0)
        
        return pred
    