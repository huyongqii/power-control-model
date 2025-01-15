import zmq
import json
from typing import Dict
import pandas as pd
from datetime import datetime

class ResourceMonitor:
    """资源使用监控器"""
    
    def __init__(self):
        self.usage_history = {}
        self.current_time = 0
        self.allocated_resources = set()
        self.total_resources = 0
        
    def on_simulation_begins(self, total_resources: int):
        """初始化监控"""
        self.usage_history = {}
        self.total_resources = total_resources
        
    def update_resource_usage(self, current_time: float, allocated_resources: set):
        """更新资源使用情况"""
        self.current_time = current_time
        self.allocated_resources = allocated_resources
        current_usage = {}
        
        for node_id in range(self.total_resources):
            is_computing = node_id in allocated_resources
            is_on = True
            cpu_usage = 1.0 if is_computing else 0.0
            
            current_usage[node_id] = {
                'cpu_usage': cpu_usage,
                'is_on': is_on,
                'is_computing': is_computing
            }
            
        self.usage_history[current_time] = current_usage
        
    def save_usage_history(self, output_file: str):
        """保存资源使用历史"""
        if not self.usage_history:
            print("Warning: No usage history to save")
            return
            
        records = []
        for timestamp, usage in self.usage_history.items():
            for node_id, node_usage in usage.items():
                records.append({
                    'timestamp': timestamp,
                    'node_id': node_id,
                    'cpu_usage': node_usage['cpu_usage'],
                    'is_on': node_usage['is_on'],
                    'is_computing': node_usage['is_computing']
                })
                
        df = pd.DataFrame(records)
        df.to_csv(output_file, index=False)
        print(f"Resource usage history saved to: {output_file}")

class SimpleScheduler:
    """简单的FCFS调度器"""
    
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.monitor = ResourceMonitor()
        self.current_time = 0
        self.allocated_resources = set()
        self.waiting_jobs = []
        
    def start(self):
        """启动调度器"""
        self.socket.connect("tcp://localhost:28000")
        
        try:
            while True:
                # 接收消息
                self.socket.send_string("{\"now\":" + str(self.current_time) + "}")
                message = self.socket.recv_string()
                data = json.loads(message)
                
                # 处理消息
                self.handle_message(data)
                
        except KeyboardInterrupt:
            print("Scheduler stopped by user")
        finally:
            self.socket.close()
            self.context.term()
            
    def handle_message(self, data):
        """处理Batsim消息"""
        events = data['events']
        for event in events:
            event_type = event['type']
            
            if event_type == "SIMULATION_BEGINS":
                self.on_simulation_begins(event)
            elif event_type == "JOB_SUBMITTED":
                self.on_job_submitted(event)
            elif event_type == "JOB_COMPLETED":
                self.on_job_completed(event)
            elif event_type == "SIMULATION_ENDS":
                self.on_simulation_ends()
                
        # 尝试调度等待的作业
        self.schedule_jobs()
        
    def on_simulation_begins(self, event):
        """模拟开始"""
        self.current_time = event['timestamp']
        resources = event['resources']
        self.monitor.on_simulation_begins(len(resources))
        self.monitor.update_resource_usage(self.current_time, self.allocated_resources)
        
    def on_simulation_ends(self):
        """模拟结束"""
        self.monitor.save_usage_history('result/resource_usage.csv')
        
    def on_job_submitted(self, event):
        """作业提交"""
        job = event['job']
        self.waiting_jobs.append(job)
        self.current_time = event['timestamp']
        self.monitor.update_resource_usage(self.current_time, self.allocated_resources)
        
    def on_job_completed(self, event):
        """作业完成"""
        job_id = event['job_id']
        # 释放资源
        for res in self.get_job_resources(job_id):
            self.allocated_resources.remove(res)
        self.current_time = event['timestamp']
        self.monitor.update_resource_usage(self.current_time, self.allocated_resources)
        
    def schedule_jobs(self):
        """调度等待的作业（FCFS策略）"""
        if not self.waiting_jobs:
            return
            
        # 按FCFS顺序处理作业
        jobs_to_remove = []
        for job in self.waiting_jobs:
            if self.can_allocate_job(job):
                self.execute_job(job)
                jobs_to_remove.append(job)
                
        # 移除已调度的作业
        for job in jobs_to_remove:
            self.waiting_jobs.remove(job)
            
    def can_allocate_job(self, job):
        """检查是否有足够的资源运行作业"""
        requested = job['requested_resources']
        available = self.get_available_resources()
        return len(available) >= requested
        
    def get_available_resources(self):
        """获取可用资源"""
        all_resources = set(range(self.monitor.total_resources))
        return all_resources - self.allocated_resources
        
    def execute_job(self, job):
        """执行作业"""
        job_id = job['id']
        requested = job['requested_resources']
        
        # 分配资源
        available = list(self.get_available_resources())
        allocated = available[:requested]
        
        # 更新已分配资源集合
        self.allocated_resources.update(allocated)
        
        # 发送执行命令
        command = {
            "timestamp": self.current_time,
            "type": "EXECUTE_JOB",
            "job_id": job_id,
            "alloc": allocated
        }
        self.socket.send_string(json.dumps(command))
        self.socket.recv_string()  # 等待确认
        
    def get_job_resources(self, job_id):
        """获取作业使用的资源"""
        # 在实际实现中，你需要维护作业到资源的映射
        # 这里简化处理，返回所有已分配的资源
        return list(self.allocated_resources)

def main():
    scheduler = SimpleScheduler()
    scheduler.start()

if __name__ == "__main__":
    main()