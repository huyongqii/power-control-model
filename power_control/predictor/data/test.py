import pandas as pd
import numpy as np

def get_time_period(hour):
    """将一天分为不同时段"""
    if 5 <= hour < 9:
        return "早晨 (5-9点)"
    elif 9 <= hour < 12:
        return "上午 (9-12点)"
    elif 12 <= hour < 14:
        return "中午 (12-14点)"
    elif 14 <= hour < 18:
        return "下午 (14-18点)"
    elif 18 <= hour < 22:
        return "晚上 (18-22点)"
    else:
        return "深夜 (22-5点)"

# 读取CSV文件
df = pd.read_csv('training_data_20250116_161848.csv')

# 将datetime列转换为datetime类型
df['datetime'] = pd.to_datetime(df['datetime'])

# 添加时间段列
df['time_period'] = df['hour'].apply(get_time_period)

# 添加工作日/周末标识
df['is_weekend'] = df['day_of_week'].apply(lambda x: '周末' if x >= 5 else '工作日')

# 计算每个时间段的平均作业数
time_period_stats = df.groupby('time_period')[['running_jobs', 'waiting_jobs']].mean()
print("\n=== 各时间段作业统计 ===")
print(time_period_stats.round(2))

# 计算工作日vs周末的平均作业数
weekday_stats = df.groupby('is_weekend')[['running_jobs', 'waiting_jobs']].mean()
print("\n=== 工作日vs周末作业统计 ===")
print(weekday_stats.round(2))

# 计算每个时间段的作业总数
time_period_total = df.groupby('time_period')[['running_jobs', 'waiting_jobs']].sum()
print("\n=== 各时间段作业总数 ===")
print(time_period_total.round(2))

# 计算工作日vs周末的作业总数
weekday_total = df.groupby('is_weekend')[['running_jobs', 'waiting_jobs']].sum()
print("\n=== 工作日vs周末作业总数 ===")
print(weekday_total.round(2))