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
df = pd.read_csv('training_data_20250118_190830.csv')

# 将datetime列转换为datetime类型
df['datetime'] = pd.to_datetime(df['datetime'])

# 从datetime列提取日期
df['date'] = df['datetime'].dt.date

# 添加时间段列
df['time_period'] = df['hour'].apply(get_time_period)

# 添加工作日/周末标识
df['is_weekend'] = df['day_of_week'].apply(lambda x: '周末' if x >= 5 else '工作日')

# 计算每个时间段的平均作业数和节点数
time_period_stats = df.groupby('time_period')[['running_jobs', 'waiting_jobs', 
                                              'nb_computing', 'nb_idle']].mean()
print("\n=== 各时间段统计（平均值）===")
print(time_period_stats.round(2))

# 计算工作日vs周末的平均作业数和节点数
weekday_stats = df.groupby('is_weekend')[['running_jobs', 'waiting_jobs',
                                         'nb_computing', 'nb_idle']].mean()
print("\n=== 工作日vs周末统计（平均值）===")
print(weekday_stats.round(2))

# 计算每个时间段的作业和节点总数
time_period_total = df.groupby('time_period')[['running_jobs', 'waiting_jobs',
                                              'nb_computing', 'nb_idle']].sum()
print("\n=== 各时间段总数 ===")
print(time_period_total.round(2))

# 计算工作日vs周末的作业和节点总数
weekday_total = df.groupby('is_weekend')[['running_jobs', 'waiting_jobs',
                                         'nb_computing', 'nb_idle']].sum()
print("\n=== 工作日vs周末总数 ===")
print(weekday_total.round(2))

# 计算每个时间段的节点使用统计
time_period_node_stats = df.groupby('time_period').agg({
    'nb_computing': ['mean', 'min', 'max'],
    'nb_idle': ['mean', 'min', 'max']
}).round(2)

# 重命名列名使其更易读
time_period_node_stats.columns = [
    'computing_平均', 'computing_最小', 'computing_最大',
    'idle_平均', 'idle_最小', 'idle_最大'
]

print("\n=== 各时间段节点使用情况 ===")
print(time_period_node_stats)

# 按时间段和日期分组，计算每天各时间段的统计
daily_period_stats = df.groupby(['date', 'time_period']).agg({
    'nb_computing': ['min', 'max']
}).round(2)

print("\n=== 每天各时间段节点使用情况 ===")
print(daily_period_stats)

# 可选：计算标准差来看波动情况
time_period_std = df.groupby('time_period').agg({
    'nb_computing': 'std',
    'nb_idle': 'std'
}).round(2)

print("\n=== 各时间段节点使用波动（标准差）===")
print(time_period_std)