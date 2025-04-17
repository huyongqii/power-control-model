#!/usr/bin/env python3
import pymysql
import csv
from datetime import datetime
from collections import defaultdict

CPUS_PER_NODE = 64

DB_CONFIG = {
    'host': 'localhost',
    'user': 'hyq',
    'password': 'hyq',
    'database': 'wm2',
    'charset': 'utf8mb4'
}

def parse_node_inx(node_str):
    """解析 node_inx 字符串为节点编号集合"""
    nodes = set()
    if not node_str:
        return nodes
    node_str = node_str.strip()
    for part in node_str.split(','):
        part = part.strip()
        if '-' in part:
            try:
                low, high = part.split('-', 1)
                low = int(low.strip())
                high = int(high.strip())
                for n in range(low, high + 1):
                    nodes.add(n)
            except Exception as e:
                print(f"解析范围错误：{part}，错误信息：{e}")
        else:
            try:
                nodes.add(int(part))
            except Exception as e:
                print(f"解析节点错误：{part}，错误信息：{e}")
    return nodes

def get_job_minutes(start, end):
    """获取作业生命周期内的每一分钟的时间戳列表"""
    minutes = []
    start_minute = ((start + 59) // 60) * 60
    end_minute = (end // 60) * 60
    for minute in range(start_minute, end_minute, 60):
        minutes.append(minute)
    return minutes

def write_to_csv(year, timeline):
    output_filename = f'wm2/job_timeline_{year}.csv'
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "datetime", 
            "running_job_count", 
            "waiting_job_count", 
            "active_node_count", 
            "avg_req_cpu_occupancy_rate",
            "avg_req_node_per_job",
            "avg_req_cpu_per_job",
            "avg_runtime_minutes"
        ])
        
        for minute_ts, stats in sorted(timeline.items()):
            if datetime.fromtimestamp(minute_ts).year == year:
                time_str = datetime.fromtimestamp(minute_ts).strftime('%Y-%m-%d %H:%M:%S')
                running = stats['running']
                waiting = stats['waiting']
                nb_computing = len(stats['active_nodes'])
                
                if nb_computing > 0:
                    avg_req_cpu_occupancy_rate = (stats['total_cpu_req'] / (nb_computing * CPUS_PER_NODE)) * 100
                else:
                    avg_req_cpu_occupancy_rate = 0.0
                
                avg_nodes_per_job = (stats['total_nodes_per_job'] / running) if running > 0 else 0
                
                avg_cpus_per_job = (stats['total_cpu_req'] / running) if running > 0 else 0
                
                avg_runtime = (stats['total_runtime'] / running / 60) if running > 0 else 0
                
                writer.writerow([
                    time_str, 
                    running, 
                    waiting, 
                    nb_computing, 
                    f"{avg_req_cpu_occupancy_rate:.2f}",
                    f"{avg_nodes_per_job:.2f}",
                    f"{avg_cpus_per_job:.2f}",
                    f"{avg_runtime:.2f}"
                ])
    
    print(f"统计结果已写入文件：{output_filename}")

def main():
    jobs = []
    conn = pymysql.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cursor:
            sql = """
            SELECT DISTINCT j.job_db_inx, j.time_submit, j.time_start, j.time_end, 
                   j.node_inx, j.cpus_req 
            FROM wm2_job_table j
            WHERE j.state = 3                  -- 成功完成的作业
              AND j.node_inx IS NOT NULL       -- 有节点信息
              AND j.node_inx != ''             -- 节点信息不为空字符串
              AND j.time_submit > 0
              AND j.time_start > j.time_submit
              AND j.time_end > j.time_start
            """
            cursor.execute(sql)
            for row in cursor.fetchall():
                job_db_inx, time_submit, time_start, time_end, node_inx, cpus_req = row
                jobs.append({
                    'job_db_inx': job_db_inx,
                    'time_submit': time_submit,
                    'time_start': time_start,
                    'time_end': time_end,
                    'node_inx': node_inx,
                    'cpus_req': cpus_req if cpus_req else 0
                })
    finally:
        conn.close()

    timeline = defaultdict(lambda: {
        'running': 0,          # 正在运行的作业数量
        'waiting': 0,          # 正在等待的作业数量
        'active_nodes': set(), # 活跃节点集合
        'total_cpu_req': 0,    # 所有运行作业请求的CPU总数
        'total_nodes_per_job': 0,  # 所有运行作业使用的节点数总和
        'total_runtime': 0,    # 所有运行作业的运行时长总和
    })
    
    total_jobs = len(jobs)
    for i, job in enumerate(jobs):
        if i % 10000 == 0:
            print(f"处理作业进度: {i}/{total_jobs} ({i/total_jobs*100:.2f}%)")
        
        job_db_inx = job['job_db_inx']
        ts_submit = job['time_submit']
        ts_start = job['time_start']
        ts_end = job['time_end']
        node_str = job['node_inx']
        cpus_req = job['cpus_req']

        if ts_start and ts_end and ts_end > 0:
            nodes = parse_node_inx(node_str)
            if nodes and cpus_req:
                nodes_count = len(nodes)
                
                for minute_ts in get_job_minutes(ts_start, ts_end):
                    # if minute_ts < ts_end:
                    timeline[minute_ts]['running'] += 1
                    timeline[minute_ts]['active_nodes'].update(nodes)
                    timeline[minute_ts]['total_cpu_req'] += cpus_req
                    timeline[minute_ts]['total_nodes_per_job'] += nodes_count
                    current_runtime = minute_ts - ts_start
                    timeline[minute_ts]['total_runtime'] += current_runtime

        if ts_submit and ts_start:
            for minute_ts in get_job_minutes(ts_submit, ts_start):
                timeline[minute_ts]['waiting'] += 1
    
    years = set()
    for minute_ts in timeline.keys():
        years.add(datetime.fromtimestamp(minute_ts).year)
    
    for year in years:
        write_to_csv(year, timeline)

if __name__ == '__main__':
    main()
