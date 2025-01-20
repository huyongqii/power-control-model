import json
import random
from datetime import datetime, timedelta
import holidays
import yaml
import sys
import os
import math
import numpy as np

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
        self.current_time = 0  # 添加current_time属性
        
    def generate_workload(self, output_file: str):
        """生成作业负载"""
        jobs = []
        job_id = 0
        
        start_date = datetime(2024, 1, 1)
        self.current_time = 0
        
        while self.current_time < self.duration_days * 24 * 3600:
            current_date = start_date + timedelta(seconds=self.current_time)
            
            # 获取当前时间点的lambda参数
            lambda_param = self._get_submit_probability(current_date)
            
            # 生成作业数量
            num_jobs = self._get_num_jobs(lambda_param)
            
            for _ in range(num_jobs):
                job = self._generate_job(job_id, self.current_time)
                jobs.append(job)
                job_id += 1
            
            # 使用指数分布生成下一个作业的时间间隔
            if self._is_working_hours(current_date):
                time_delta = int(np.random.exponential(scale=180))  # 平均3分钟
                time_delta = max(30, min(time_delta, 600))  # 限制在30秒到10分钟之间
            else:
                time_delta = int(np.random.exponential(scale=600))  # 平均10分钟
                time_delta = max(180, min(time_delta, 1800))  # 限制在3分钟到30分钟之间
            
            self.current_time += time_delta
        
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
        """获取特定时间点提交作业的概率
        
        使用基准概率结合时间因素计算lambda参数：
        - 深夜(24:00-5:00): 极低概率 (基准概率的5%)
        - 早晨(5:00-9:00): 较低概率 (基准概率的30%)
        - 上午(9:00-12:00): 高峰期 (基准概率的100%)
        - 中午(12:00-14:00): 较低概率 (基准概率的60%)
        - 下午(14:00-18:00): 高峰期 (基准概率的100%)
        - 晚上(18:00-24:00): 中等概率 (基准概率的80%)
        """
        base_lambda = self.base_submit_prob * 5  # 将基准概率转换为泊松分布的lambda参数
        
        # 节假日降低概率
        if current_date.date() in self.cn_holidays:
            base_lambda *= 0.3
        # 周末降低概率
        elif current_date.weekday() >= 5:  # 5=周六，6=周日
            base_lambda *= 0.4
            
        # 根据小时调整lambda参数
        hour = current_date.hour
        if hour < 5:  # 深夜(0:00-5:00)
            base_lambda *= 0.05
        elif 5 <= hour < 9:  # 早晨
            base_lambda *= 0.3
        elif 9 <= hour < 12:  # 上午(高峰)
            base_lambda *= 1.0
        elif 12 <= hour < 14:  # 中午
            base_lambda *= 0.6
        elif 14 <= hour < 18:  # 下午(高峰)
            base_lambda *= 1.0
        elif 18 <= hour < 24:  # 晚上
            base_lambda *= 0.8
            
        return base_lambda
    
    def _get_num_jobs(self, lambda_param: float) -> int:
        """使用泊松分布生成当前时间点提交的作业数量
        
        Args:
            lambda_param: 泊松分布的lambda参数
        
        Returns:
            int: 生成的作业数量
        """
        # 使用泊松分布生成作业数量，并限制最大数量
        num_jobs = np.random.poisson(lambda_param)
        return min(num_jobs, 5)  # 限制单个时间点最多提交5个作业
    
    def _is_working_hours(self, current_date: datetime) -> bool:
        """判断是否为工作时间（扩大工作时间范围）"""
        if current_date.weekday() >= 5:  # 周末
            return 9 <= current_date.hour < 18  # 周末工作时间短一些
        if current_date.date() in self.cn_holidays:  # 节假日
            return 9 <= current_date.hour < 18
        return 7 <= current_date.hour < 22  # 工作日扩大工作时间范围
        
    def _generate_job(self, job_id: int, submit_time: int) -> dict:
        """生成单个作业
        
        根据实际HPC环境的特点调整资源分配和运行时间:
        - 短作业(40%): 运行时间较短，通常是测试或小型计算
        - 中等作业(40%): 正常的计算任务
        - 长作业(20%): 大型计算任务，运行时间较长
        """
        rand = random.random()
        
        # 首先决定作业类型和运行时间
        if rand < 0.4:  # 短作业
            walltime = random.randint(1800, 7200)  # 30分钟到2小时
        elif rand < 0.8:  # 中等作业
            walltime = random.randint(7200, 43200)  # 2小时到12小时
        else:  # 长作业
            walltime = random.randint(43200, 172800)  # 12小时到48小时
        
        # 然后决定资源需求
        resource_rand = random.random()
        if resource_rand < 0.8:  # 80%的作业
            requested_resources = 1
        elif resource_rand < 0.95:  # 15%的作业
            requested_resources = random.randint(2, 4)
        elif resource_rand < 0.999:  # 4.9%的作业
            requested_resources = random.randint(5, 20)
        else:  # 0.1%的大规模作业
            requested_resources = random.randint(21, min(60, self.total_nodes))
        
        return {
            "id": f"job_{job_id}",
            "subtime": submit_time,
            "res": requested_resources,
            "profile": f"profile_{job_id}",
            "walltime": walltime
        }
        
    def _generate_profiles(self, jobs: list) -> dict:
        """生成作业配置文件
        
        根据节点配置(56核，每核40GF)调整计算负载：
        - 总计算能力 = 56核 * 40GF = 2240 GF/节点
        - 考虑实际使用率和负载分布
        """
        profiles = {}
        node_peak_flops = 56 * 40e9  # 每个节点的峰值计算能力（56核 * 40 GF）
        
        for job in jobs:
            job_id = job["id"]
            walltime = job["walltime"]
            requested_resources = job["res"]
            
            # 根据作业运行时间调整负载强度
            if walltime < 7200:  # 短作业
                cpu_util = random.uniform(0.1, 0.15)  # 进一步降低短作业的CPU利用率
            elif walltime < 43200:  # 中等作业
                cpu_util = random.uniform(0.12, 0.2)  # 调整中等作业的CPU利用率范围
            else:  # 长作业
                cpu_util = random.uniform(0.15, 0.25)  # 适当降低长作业的CPU利用率上限
            
            # 计算每秒的计算负载（考虑多节点的情况）
            # 对于多节点作业，考虑并行效率损失
            parallel_efficiency = 1.0 if requested_resources == 1 else (0.9 - 0.1 * math.log10(requested_resources))
            flops_per_second = (node_peak_flops * 0.01 * cpu_util * parallel_efficiency)
            
            # 总计算量需要考虑实际执行时间，而不是walltime
            # 为长作业预留更多余量
            if walltime > 43200:  # 12小时以上的作业
                runtime_ratio = random.uniform(0.4, 0.7)  # 给长作业更多余量
            else:
                runtime_ratio = random.uniform(0.5, 0.8)
            actual_runtime = int(walltime * runtime_ratio)
            
            total_flops = flops_per_second * actual_runtime
            
            # 通信量与节点数和运行时间相关
            if requested_resources > 1:
                # 多节点作业的通信量，考虑节点数的对数增长
                comm_ratio = random.uniform(0.005, 0.02) * (1 + math.log(requested_resources) / 15)
                comm_amount = flops_per_second * comm_ratio
            else:
                # 单节点作业通信量很小
                comm_amount = flops_per_second * random.uniform(0.0005, 0.002)
            
            profiles[f"profile_{job_id.split('_')[1]}"] = {
                "type": "parallel_homogeneous",
                "cpu": total_flops,
                "com": comm_amount
            }
        
        return profiles

def main():
    # 生成作业数据
    generator = WorkloadGenerator()
    generator.generate_workload(os.path.join(OUTPUT_DIR, 'jobs.json'))

if __name__ == "__main__":
    main()
