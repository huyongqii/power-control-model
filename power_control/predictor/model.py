import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """注意力层"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # 加权求和
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class NodePredictorNN(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        
        # LSTM配置
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 注意力层
        self.attention = AttentionLayer(self.lstm_hidden_size * 2)  # 双向LSTM输出维度翻倍
        
        # 时间特征处理 (6个时间特征)
        self.time_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2)
        )
        
        self.dayback_fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2)
        )
        
        # 计算组合特征的总维度
        # LSTM输出: lstm_hidden_size * 2 (双向) = 256
        # 时间特征: 32
        # 历史同期特征: 16
        combined_input_size = 256 + 32 + 16
        
        # 组合特征的全连接层
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 2)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x_past_hour, x_cur_datetime, x_dayback):
        batch_size = x_past_hour.size(0)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x_past_hour)
        
        # 应用注意力机制
        lstm_out, attention_weights = self.attention(lstm_out)
        
        # 处理时间特征
        time_out = self.time_fc(x_cur_datetime)
        
        # 处理历史同期数据
        dayback_out = self.dayback_fc(x_dayback)
        
        # 组合所有特征
        combined = torch.cat([lstm_out, time_out, dayback_out], dim=1)
        
        return self.combined_fc(combined)
    