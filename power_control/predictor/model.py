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

        # LSTM层
        self.lstm_hidden_size = 128  # 减少到128
        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,  # 减少到2层
            batch_first=True,
            dropout=0.2,  # 略微减少dropout
            bidirectional=True
        )
        
        # 时间特征处理增强
        self.time_fc = nn.Sequential(
            nn.Linear(6, 32),  # 减少到32
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2)
        )
        
        self.dayback_fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Dropout(0.2)
        )
        
        # 调整组合特征的维度
        combined_input_size = 256 + 16 + 8  # LSTM(128*2) + time(16) + dayback(8)
        
        # 更深的全连接层
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 2)
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
        # LSTM处理
        lstm_out, _ = self.lstm(x_past_hour)
        
        # 全局平均池化
        lstm_out = torch.mean(lstm_out, dim=1)
        
        # 处理其他特征
        time_out = self.time_fc(x_cur_datetime)
        dayback_out = self.dayback_fc(x_dayback)
        
        # 组合所有特征
        combined = torch.cat([lstm_out, time_out, dayback_out], dim=1)
        
        output = self.combined_fc(combined)
        return output
    