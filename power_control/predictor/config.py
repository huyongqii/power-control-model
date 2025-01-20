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

# 模型配置
MODEL_CONFIG = {
    'epochs': 100,
    'batch_size': 128,
    'learning_rate': 0.001,
    'min_lr': 1e-6,
    'lr_patience': 5,
    'lr_factor': 0.5,
    'early_stopping_patience': 15,
    'lookback_minutes': 4 * 60,
    'forecast_minutes': 30,
    'model_dir': MODEL_DIR,
    'data_path': os.path.join(DATA_DIR, 'training_data_20250120_190101.csv'),
    'log_dir': LOG_DIR,
    'grad_clip': 1.0
}