import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime, timedelta

class DataProcessor:
    """处理Batsim模拟数据，生成训练数据集"""
    def __init__(self):
        self.start_date = datetime(2024, 1, 1)  # 设置起始时间
        
    def process_simulation_data(self, result_dir: str) -> pd.DataFrame:
        """处理模拟结果数据"""
        result_path = Path(result_dir)
        
        data_dir = Path('/root/PredictModel/hpc_simulation/power_control/data/processed_data')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取原始数据
        machine_states = pd.read_csv(result_path / 'out_machine_states.csv')
        jobs_data = pd.read_csv(result_path / 'out_jobs.csv')
        energy_data = pd.read_csv(result_path / 'out_consumed_energy.csv')
        
        # 使用相对于start_date的时间
        machine_states['timestamp'] = pd.to_datetime(
            machine_states['time'].apply(
                lambda x: self.start_date + timedelta(seconds=float(x))
            )
        )
        result_df = machine_states.set_index('timestamp')
        
        # 添加节点状态信息
        result_df['total_nodes'] = (result_df['nb_computing'] + result_df['nb_idle'] + 
                                  result_df['nb_sleeping'] + result_df['nb_switching_on'] + 
                                  result_df['nb_switching_off'])
        result_df['active_nodes'] = result_df['nb_computing'] + result_df['nb_idle']
        result_df['utilization_rate'] = result_df['nb_computing'] / result_df['total_nodes']
        
        # 添加作业信息和能耗信息
        result_df = self._add_job_info(result_df, jobs_data)
        result_df = self._add_energy_info(result_df, energy_data)
        
        # 保存处理后的数据
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = data_dir / f'training_data_{timestamp}.csv'
        result_df.to_csv(output_path)
        print(f"Processed data saved to: {output_path}")
        
        return result_df
        
    def _add_job_info(self, df: pd.DataFrame, jobs_data: pd.DataFrame) -> pd.DataFrame:
        """添加作业信息"""
        # 转换作业时间
        jobs_data['submission_time'] = pd.to_datetime(
            jobs_data['submission_time'].apply(
                lambda x: self.start_date + timedelta(seconds=float(x))
            )
        )
        jobs_data['starting_time'] = pd.to_datetime(
            jobs_data['starting_time'].apply(
                lambda x: self.start_date + timedelta(seconds=float(x))
            )
        )
        jobs_data['finish_time'] = pd.to_datetime(
            jobs_data['finish_time'].apply(
                lambda x: self.start_date + timedelta(seconds=float(x))
            )
        )
        
        # 计算每个时间点的作业状态
        running_jobs = []
        waiting_jobs = []
        
        for timestamp in df.index:
            running = len(jobs_data[(jobs_data['starting_time'] <= timestamp) & (jobs_data['finish_time'] > timestamp)])
            waiting = len(jobs_data[(jobs_data['submission_time'] <= timestamp) & (jobs_data['starting_time'] > timestamp)])
            running_jobs.append(running)
            waiting_jobs.append(waiting)
            
        df['running_jobs'] = running_jobs
        df['waiting_jobs'] = waiting_jobs
        
        return df
        
    def _add_energy_info(self, df: pd.DataFrame, energy_data: pd.DataFrame) -> pd.DataFrame:
        """添加能耗信息"""
        energy_data['timestamp'] = pd.to_datetime(
            energy_data['time'].apply(
                lambda x: self.start_date + timedelta(seconds=float(x))
            )
        )
        
        energy_data = energy_data[energy_data['event_type'] == 's']
        energy_data = energy_data.set_index('timestamp')
        
        # 只保留需要的列
        df = df.join(energy_data[['energy', 'wattmin', 'epower']], how='left')
        
        # 使用前向填充处理空值
        df[['energy', 'wattmin', 'epower']] = df[['energy', 'wattmin', 'epower']].fillna(method='ffill')
        
        return df
    
def process_simulation_results():
    # 初始化处理器
    processor = DataProcessor()
    
    # 处理数据
    training_data = processor.process_simulation_data('/root/PredictModel/hpc_simulation/result')
    
    print("Data processing completed!")
    print("\nDataset shape:", training_data.shape)
    print("\nFeatures:")
    print(training_data.columns.tolist())
    print("\nSample data:")
    print(training_data.head())
    
    return training_data

if __name__ == "__main__":
    process_simulation_results()