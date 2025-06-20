import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'train', 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_CONFIG = {
    'epochs': 100,
    'batch_size': 256,
    'learning_rate': 0.001,
    'min_lr': 1e-6,
    'lr_patience': 5,
    'lr_factor': 0.5,
    'early_stopping_patience': 15,
    'weight_decay': 0.0001,
    'gradient_clip': 1.0,
    'warmup_epochs': 5,
    'lookback_minutes': 240,
    'forecast_minutes': 30,
    'model_dir': MODEL_DIR,
    # 'data_dir': "/home/hyq/green-energy/sk1",
    # 'data_path': os.path.join(DATA_DIR, 'training_data_20250120_190101.csv'),
    # 'data_path': "/home/hyq/green-energy/sk1/combined_job_timeline.csv",
    'total_nodes': 120,
    'data_path': "/home/hyq/green-energy/sk1/job_timeline_2023.csv",
    'log_dir': LOG_DIR,
    'grad_clip': 1.0,
    'scaler_path': "/home/hyq/green-energy/sk1/job_timeline_2024/dataset_scalers.pkl"
}