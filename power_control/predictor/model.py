import torch
import torch.nn as nn
import torch.nn.functional as F

class NodePredictorNN(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        
        # 1. 历史序列编码器
        self.past_encoder = nn.Sequential(
            # 1D卷积层处理时序特征
            nn.Conv1d(feature_size, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            # 自注意力机制处理长序列依赖
            SelfAttention(256),
            # 全局池化
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
        
        # 融合层保持不变
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128 + 128, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        
        # 修改预测头，添加离散化处理
        self.predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            # 添加激活函数限制输出范围
            nn.Softplus()  # 确保输出非负且数值稳定
        )
        
    def forward(self, past_hour, cur_datetime, dayback):
        # 处理历史序列
        past_hour = past_hour.transpose(1, 2)
        past_features = self.past_encoder(past_hour).squeeze(-1)
        
        # 处理时间特征
        datetime_features = self.datetime_encoder(cur_datetime)
        
        # 处理历史模式特征
        dayback_features = self.dayback_encoder(dayback)
        
        # 特征融合
        combined = torch.cat([past_features, datetime_features, dayback_features], dim=1)
        fused = self.fusion(combined)
        
        # 获取基础预测值
        base_pred = self.predictor(fused)
        
        # 限制预测范围，增加数值稳定性
        pred = torch.clamp(base_pred, min=0.0, max=1000.0)
        
        return pred

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        B, C, L = x.size()
        
        # 生成查询、键、值
        proj_query = self.query(x).view(B, -1, L).permute(0, 2, 1)  # (B, L, C)
        proj_key = self.key(x).view(B, -1, L)  # (B, C, L)
        proj_value = self.value(x).view(B, -1, L)  # (B, C, L)
        
        # 计算注意力权重
        energy = torch.bmm(proj_query, proj_key)  # (B, L, L)
        attention = self.softmax(energy)  # (B, L, L)
        
        # 应用注意力
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # (B, C, L)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out
    