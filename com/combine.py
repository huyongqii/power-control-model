import pandas as pd
import os

# 定义要合并的文件列表
files = [
    'job_timeline_2024.csv',
    'job_timeline_2023.csv'
]

# 创建一个空列表来存储每个文件的数据
dfs = []

# 读取每个CSV文件并添加到列表中
for file in files:
    if os.path.exists(file):  # 检查文件是否存在
        df = pd.read_csv(file)
        dfs.append(df)
    else:
        print(f"警告: 文件 {file} 不存在")

# 如果有数据要合并
if dfs:
    # 垂直合并所有数据框
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 保存合并后的数据到新的CSV文件
    combined_df.to_csv('combined_job_timeline.csv', index=False)
    print("文件合并完成！输出文件名: combined_job_timeline.csv")
else:
    print("没有找到任何可以合并的文件")