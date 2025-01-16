import json
import random
from datetime import datetime, timedelta
import holidays
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR,'hpc_env', 'data')
CONFIG_DIR = os.path.join(BASE_DIR,'hpc_env', 'platforms')

def load_config(config_file='config.yaml'):
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class WorkloadGenerator:
    """生成真实场景的HPC作业负载"""
    
    def __init__(self, config_file=os.path.join(CONFIG_DIR, 'config.yaml')):
        config = load_config(config_file)
        self.total_nodes = config['cluster']['num_nodes']
        self.duration_days = config['workload']['duration_days']
        self.base_submit_prob = config['workload']['base_submit_prob']
        self.cn_holidays = holidays.CN()  # 中国节假日
        
    def generate_workload(self, output_file: str):
        """生成作业负载"""
        jobs = []
        job_id = 0
        
        start_date = datetime(2024, 1, 1)  # 从2024年1月1日开始
        current_time = 0  # batsim的模拟时间（秒）
        
        while current_time < self.duration_days * 24 * 3600:
            # 获取当前时间点
            current_date = start_date + timedelta(seconds=current_time)
            
            # 计算提交作业的概率
            submit_prob = self._get_submit_probability(current_date)
            
            # 可能在同一时间点提交多个作业
            num_jobs = self._get_num_jobs(submit_prob)
            
            for _ in range(num_jobs):
                job = self._generate_job(job_id, current_time)
                jobs.append(job)
                job_id += 1
            
            # 缩短时间间隔
            if self._is_working_hours(current_date):
                current_time += random.randint(30, 180)  # 30秒到3分钟
            else:
                current_time += random.randint(180, 600)  # 3分钟到10分钟
        
        # 保存为batsim工作负载格式
        workload = {
            "jobs": jobs,
            "nb_res": self.total_nodes,
            "profiles": self._generate_profiles(jobs)
        }
        
        with open(output_file, 'w') as f:
            json.dump(workload, f, indent=2)
            
        print(f"Generated {len(jobs)} jobs for {self.duration_days} days")
        
    def _get_submit_probability(self, current_date: datetime) -> float:
        """获取特定时间点提交作业的概率"""
        base_prob = self.base_submit_prob
        
        # 节假日降低概率但不要太低
        if current_date.date() in self.cn_holidays:
            base_prob *= 0.5
        # 周末降低概率但不要太低
        elif current_date.weekday() >= 5:  # 5=周六，6=周日
            base_prob *= 0.6
            
        # 根据小时调整概率
        hour = current_date.hour
        if 0 <= hour < 6:  # 凌晨
            base_prob *= 0.4
        elif 6 <= hour < 9:  # 早晨
            base_prob *= 0.8
        elif 9 <= hour < 17:  # 工作时间
            base_prob *= 1.0
        elif 17 <= hour < 20:  # 傍晚
            base_prob *= 0.9
        else:  # 晚上
            base_prob *= 0.6
            
        return base_prob
    
    def _get_num_jobs(self, prob: float) -> int:
        """根据概率决定在当前时间点提交的作业数量"""
        if random.random() > prob:
            return 0
            
        # 在工作时间可能同时提交多个作业
        weights = [0.5, 0.3, 0.15, 0.05]  # 权重分别对应提交1,2,3,4个作业的概率
        return random.choices(range(1, 5), weights=weights)[0]
    
    def _is_working_hours(self, current_date: datetime) -> bool:
        """判断是否为工作时间（扩大工作时间范围）"""
        if current_date.weekday() >= 5:  # 周末
            return 9 <= current_date.hour < 18  # 周末工作时间短一些
        if current_date.date() in self.cn_holidays:  # 节假日
            return 9 <= current_date.hour < 18
        return 7 <= current_date.hour < 22  # 工作日扩大工作时间范围
        
    def _generate_job(self, job_id: int, submit_time: int) -> dict:
        """生成单个作业"""
        # # 直接使用传入的datetime对象
        # submit_time_str = submit_datetime.isoformat()
        
        # 生成资源请求分布逻辑保持不变
        if random.random() < 0.7:
            requested_resources = random.randint(1, 20)
            walltime = random.randint(300, 7200)
        elif random.random() < 0.9:
            requested_resources = random.randint(21, 60)
            walltime = random.randint(7200, 43200)
        else:
            requested_resources = random.randint(61, self.total_nodes)
            walltime = random.randint(43200, 86400)
            
        return {
            "id": f"job_{job_id}",
            "subtime": submit_time,
            "res": requested_resources,
            "profile": f"profile_{job_id}",
            "walltime": walltime
        }
        
    def _generate_profiles(self, jobs: list) -> dict:
        """生成作业配置文件"""
        profiles = {}
        for job in jobs:
            job_id = job["id"]
            walltime = job["walltime"]
            requested_resources = job["res"]
            
            # 根据请求的资源数调整计算量
            # total_computing_power = 20e9 * requested_resources  # 每个节点40 GFLOPS
            
            # 设置作业的计算负载（使用20%-80%的可用计算能力）
            flops_per_second = random.uniform(0.2, 0.8) * 1e12 * requested_resources
            total_flops = flops_per_second * walltime
            
            # 通信量设置为计算量的1%-5%
            comm_ratio = random.uniform(0.01, 0.05)
            comm_amount = flops_per_second * comm_ratio
            
            profiles[f"profile_{job_id.split('_')[1]}"] = {
                "type": "parallel_homogeneous",
                "cpu": flops_per_second,
                "com": comm_amount
            }
        
        return profiles

def main():
    # 生成作业数据
    generator = WorkloadGenerator()
    generator.generate_workload(os.path.join(OUTPUT_DIR, 'jobs.json'))

if __name__ == "__main__":
    main()
