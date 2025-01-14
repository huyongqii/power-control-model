import json
import random
from datetime import datetime, timedelta
import holidays

class WorkloadGenerator:
    """生成真实场景的HPC作业负载"""
    
    def __init__(self, total_nodes=100, duration_days=30):
        self.total_nodes = total_nodes
        self.duration_days = duration_days
        self.cn_holidays = holidays.CN()  # 中国节假日
        
    def generate_workload(self, output_file: str):
        """生成作业负载"""
        jobs = []
        job_id = 0
        
        # 生成一个月的数据
        start_date = datetime(2024, 1, 1)  # 从2024年1月1日开始
        current_time = 0  # batsim的模拟时间（秒）
        
        while current_time < self.duration_days * 24 * 3600:
            # 获取当前时间点
            current_date = start_date + timedelta(seconds=current_time)
            
            # 计算提交作业的概率
            submit_prob = self._get_submit_probability(current_date)
            
            # 根据概率决定是否提交作业
            if random.random() < submit_prob:
                # 直接传入current_time而不是current_date
                job = self._generate_job(job_id, current_time)
                jobs.append(job)
                job_id += 1
            
            # 时间前进（随机间隔，但工作时间间隔较短）
            if self._is_working_hours(current_date):
                current_time += random.randint(60, 300)  # 1-5分钟
            else:
                current_time += random.randint(300, 900)  # 5-15分钟
        
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
        base_prob = 0.6  # 基础概率
        
        # 节假日降低概率
        if current_date.date() in self.cn_holidays:
            base_prob *= 0.3
        # 周末降低概率
        elif current_date.weekday() >= 5:  # 5=周六，6=周日
            base_prob *= 0.4
            
        # 根据小时调整概率
        hour = current_date.hour
        if 0 <= hour < 6:  # 凌晨
            base_prob *= 0.2
        elif 6 <= hour < 9:  # 早晨
            base_prob *= 0.7
        elif 9 <= hour < 17:  # 工作时间
            base_prob *= 1.0
        elif 17 <= hour < 20:  # 傍晚
            base_prob *= 0.8
        else:  # 晚上
            base_prob *= 0.4
            
        return base_prob
        
    def _is_working_hours(self, current_date: datetime) -> bool:
        """判断是否为工作时间"""
        if current_date.weekday() >= 5:  # 周末
            return False
        if current_date.date() in self.cn_holidays:  # 节假日
            return False
        return 9 <= current_date.hour < 17  # 工作时间
        
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
            requested_resources = job["res"]
            
            # 为每个请求的节点生成随机FLOPS
            flops_per_node = [random.uniform(1e6, 1e7) for _ in range(requested_resources)]
            
            # 生成通信矩阵
            com_matrix = []
            for i in range(requested_resources):
                row = [random.uniform(0, 1e6) if j <= i else 0 for j in range(requested_resources)]
                com_matrix.extend(row)
            
            profiles[f"profile_{job_id.split('_')[1]}"] = {
                "type": "parallel",
                "cpu": flops_per_node,
                "com": com_matrix
            }
            
        return profiles

def main():
    # 生成一个月的数据
    generator = WorkloadGenerator(total_nodes=500, duration_days=30)
    generator.generate_workload("data/jobs.json")

if __name__ == "__main__":
    main()
