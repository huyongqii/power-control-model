import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime, timedelta
from multiprocessing import Pool
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(BASE_DIR,'hpc_env', 'data', 'result')
OUTPUT_DIR = os.path.join(BASE_DIR,'power_control', 'predictor', 'data')

# 设置起始时间
start_time = datetime(2025, 1, 1)

def count_running_jobs(args):
    """计算指定时间点正在运行的作业数"""
    time, jobs_data = args
    return len(jobs_data[
        (jobs_data['starting_time'] <= time) & 
        (jobs_data['finish_time'] > time)
    ])

def count_waiting_jobs(args):
    """计算指定时间点等待中的作业数"""
    time, jobs_data = args
    return len(jobs_data[
        (jobs_data['submission_time'] <= time) & 
        (jobs_data['starting_time'] > time)
    ])

def process_simulation_data(result_dir: str) -> pd.DataFrame:
    """处理模拟结果数据"""
    result_path = Path(result_dir)
    
    # 读取数据
    machine_states = pd.read_csv(result_path / 'out_machine_states.csv')
    energy_data = pd.read_csv(result_path / 'out_consumed_energy.csv')
    jobs_data = pd.read_csv(result_path / 'out_jobs.csv')
    
    # 1. 处理机器状态数据
    machine_states = machine_states.rename(columns={'time': 'timestamp'})
    machine_states = machine_states.drop_duplicates(subset=['timestamp'], keep='last')
    machine_states = machine_states.set_index('timestamp')
    
    # 2. 处理能耗数据
    energy_data = energy_data.rename(columns={'time': 'timestamp'})
    energy_data = energy_data.drop('event_type', axis=1)
    energy_data = energy_data.drop_duplicates(subset=['timestamp'], keep='last')
    energy_data = energy_data.set_index('timestamp')
    
    # 3. 处理作业数据
    all_times = pd.Series(name='timestamp')
    
    # 添加作业提交、开始和结束时间
    all_times = pd.concat([
        all_times,
        pd.Series(jobs_data['submission_time']),
        pd.Series(jobs_data['starting_time']),
        pd.Series(jobs_data['finish_time'])
    ]).drop_duplicates().sort_values()
    
    # 创建基础时间序列数据框
    df = pd.DataFrame(index=all_times)
    df.index.name = 'timestamp'
    
    # 4. 首先添加时间特征
    df['datetime'] = df.index.map(lambda x: start_time + timedelta(seconds=float(x)))
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # 5. 计算作业数量
    time_jobs_pairs = [(time, jobs_data) for time in df.index]
    
    with Pool() as pool:
        df['running_jobs'] = pool.map(count_running_jobs, time_jobs_pairs)
        df['waiting_jobs'] = pool.map(count_waiting_jobs, time_jobs_pairs)
    
    # 6. 合并数据
    df = df.join(machine_states, how='left')
    df[machine_states.columns] = df[machine_states.columns].fillna(method='ffill')
    
    df = df.join(energy_data, how='left')
    df[energy_data.columns] = df[energy_data.columns].fillna(method='ffill')
    
    # 7. 计算利用率
    df['utilization_rate'] = df['nb_computing'] / (df['nb_computing'] + df['nb_idle'])
    
    # 8. 确保所有数值都有效
    df = df.fillna(0)
    
    # 9. 重新排列列的顺序，确保时间字段在最前面
    columns_order = ['datetime', 'hour', 'day_of_week'] + [col for col in df.columns if col not in ['datetime', 'hour', 'day_of_week']]
    df = df[columns_order]
    
    # 在第8步之后，添加时间规整化处理
    def round_to_nearest_minute(timestamp):
        """将时间戳舍入到最近的整分钟"""
        seconds = float(timestamp)
        rounded_seconds = round(seconds / 60) * 60
        return rounded_seconds

    # 添加是否为原始数据的标志位
    df['is_original'] = 1
    
    # 将时间戳规整到最近的分钟
    df.index = df.index.map(round_to_nearest_minute)
    
    # 如果规整后出现重复的时间戳，保留第一条记录
    df = df[~df.index.duplicated(keep='first')]
    
    # 创建完整的分钟级时间序列
    min_time = float(df.index.min())  # 转换为 float 类型
    max_time = float(df.index.max())  # 转换为 float 类型
    
    # 使用 numpy 创建时间序列
    time_range = np.arange(
        start=min_time,
        stop=max_time + 60,  # 加60秒确保包含最后一分钟
        step=60  # 每60秒一个数据点
    )
    
    # 创建新的数据框
    new_df = pd.DataFrame(index=time_range)
    new_df.index.name = 'timestamp'
    
    # 使用原有数据进行插值
    interpolated_df = df.reindex(new_df.index, method='ffill')
    
    # 对插值的数据设置is_original为0
    interpolated_df.loc[~interpolated_df.index.isin(df.index), 'is_original'] = 0
    
    # 重新计算datetime列
    interpolated_df['datetime'] = interpolated_df.index.map(lambda x: start_time + timedelta(seconds=float(x)))
    interpolated_df['hour'] = interpolated_df['datetime'].dt.hour
    interpolated_df['day_of_week'] = interpolated_df['datetime'].dt.dayofweek
    
    # 重新排列列的顺序
    priority_columns = [
        'is_original', 'datetime', 'hour', 'day_of_week',
        'nb_computing', 'nb_idle', 'utilization_rate',
        'running_jobs', 'waiting_jobs',
        'wattmin', 'epower', 'energy'
    ]
    
    # 获取其他列
    other_columns = [col for col in interpolated_df.columns if col not in priority_columns]
    
    # 按照指定顺序重新排列列
    final_df = interpolated_df[priority_columns + other_columns]
    
    return final_df

def save_processed_data(df: pd.DataFrame, output_dir: str):
    """保存处理后的数据"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 确保所有列都是数值类型
    numeric_columns = ['nb_sleeping', 'nb_switching_on', 'nb_switching_off', 
                      'nb_idle', 'nb_computing', 'energy', 'wattmin', 
                      'epower', 'running_jobs', 'waiting_jobs', 'utilization_rate']
    
    # 转换数据类型
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 生成输出文件名（包含时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_path / f'training_data_{timestamp}.csv'
    
    # 保存数据
    df.to_csv(output_file)
    print(f"Saved processed data to: {output_file}")
    
    # 打印基本信息
    print("\nData shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    
    # 分别打印每个数值列的统计信息
    print("\nNumerical columns statistics:")
    for col in numeric_columns:
        if col in df.columns:
            print(f"\n{col}:")
            stats = df[col].describe()
            print(f"  count: {stats['count']:.2f}")
            print(f"  mean:  {stats['mean']:.2f}")
            print(f"  std:   {stats['std']:.2f}")
            print(f"  min:   {stats['min']:.2f}")
            print(f"  25%:   {stats['25%']:.2f}")
            print(f"  50%:   {stats['50%']:.2f}")
            print(f"  75%:   {stats['75%']:.2f}")
            print(f"  max:   {stats['max']:.2f}")
    
    # 检查缺失值
    print("\nMissing values:")
    print(df.isnull().sum())

def process_simulation_results():
    """主处理函数"""
    # 确保输出目录存在
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 处理数据
    print("Processing simulation data...")
    training_data = process_simulation_data(INPUT_DIR)
    
    # 保存处理后的数据
    print("\nSaving processed data...")
    save_processed_data(training_data, OUTPUT_DIR)
    
    # 打印样本数据
    print("\nSample data (first 5 rows):")
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(training_data.head())
    
    return training_data

if __name__ == "__main__":
    process_simulation_results()