import os

# 设置基础目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'power_control', 'predictor', 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'power_control', 'predictor', 'models')
LOG_DIR = os.path.join(BASE_DIR, 'power_control', 'predictor', 'logs')

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 模型配置，使用绝对路径
MODEL_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    'lookback_minutes': 24*60,
    'forecast_minutes': 30,
    'model_dir': MODEL_DIR,  # 修正为绝对路径
    'data_path': os.path.join(DATA_DIR, 'training_data_20250116_161848.csv'),  # 修正为绝对路径
    'log_dir': LOG_DIR,  # 修正为绝对路径
}