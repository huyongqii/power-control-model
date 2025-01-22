from batsim.batsim import BatsimScheduler
from procset import ProcSet
from powerControl import NodePowerController

class PowerControlScheduler(BatsimScheduler):
    """实现了FCFS调度算法并集成了电源控制的调度器"""
    
    def __init__(self, options=None):
        super().__init__(options)
        self.power_controller = None
        self.power_check_interval = 1800  # 每30分钟检查一次电源状态
        self.record_interval = 60         # 用于设置下一次回调时间
        self.last_power_check = 0         # 上次电源检查时间
        
        # 添加能耗相关属性
        self.last_energy = 0.0           # 上次能耗值
        self.last_energy_time = 0.0      # 上次能耗时间
        self.current_power = 0.0         # 当前功率
        
        self.waiting_jobs = []           # 等待队列
        self.running_jobs = []           # 运行中的作业

    def onAfterBatsimInit(self):
        """初始化调度器"""
        super().onAfterBatsimInit()
        
    def onSimulationBegins(self):
        super().onSimulationBegins()
        """模拟开始时的回调"""
        print("Simulation begins at time:", self.bs.time())
        super().onSimulationBegins()
        
        # 初始化电源控制器
        self.power_controller = NodePowerController(batsim_scheduler=self.bs)
        
        # 初始化时间记录
        self.last_power_check = self.bs.time()
        
        # 设置第一次回调
        self.schedule_next_record()

    def schedule_next_record(self):
        """设置下一次回调的时间（每分钟）"""
        next_check = self.bs.time() + self.record_interval  # 1分钟后
        self.bs.wake_me_up_at(next_check)

    def onRequestedCall(self):
        """响应定时回调"""
        print("onRequestedCall is called at time:", self.bs.time())
        current_time = self.bs.time()
        
        # 检查是否需要执行电源管理（每30分钟）
        if current_time >= self.last_power_check + self.power_check_interval:
            self.power_controller.ExecutePowerActions()
            self.last_power_check = current_time
        
        # 请求能耗数据并记录系统状态（每次回调都执行）
        self.bs.request_consumed_energy()
        self.power_controller.RecordSystemState(
            current_time,
            self.running_jobs,
            self.waiting_jobs,
            self.current_power
        )
        
        # 设置下一次回调时间
        self.schedule_next_record()

    def onJobSubmission(self, job):
        """处理作业提交"""
        if job.requested_resources > self.bs.nb_resources:
            self.bs.reject_jobs([job])
            return
            
        self.waiting_jobs.append(job)
        self.try_schedule_jobs()

    def onJobCompletion(self, job):
        """处理作业完成"""
        if job in self.running_jobs:
            self.running_jobs.remove(job)
            
        print("onJobCompletion job.allocation {} for job: {}".format(job.allocation, job.id))
        for node in job.allocation:
            self.power_controller.RemoveJobFromNode(node)
            
        # 尝试调度等待的作业
        self.try_schedule_jobs()

    def onJobKilled(self, job):
        """处理作业被杀死"""
        if job in self.running_jobs:
            self.running_jobs.remove(job)
        for node in job.allocation:
            self.power_controller.RemoveJobFromNode(node)

    def onJobMessage(self, timestamp, job, message):
        """处理来自作业的消息"""
        print(f"Message from job {job.id} at time {timestamp}: {message}")

    def onMachinePStateChanged(self, machines, new_pstate):
        """处理节点电源状态变化"""
        for machine in machines:
            self.power_controller.HandleStateTransitionComplete(machine, new_pstate)

    def onReportEnergyConsumed(self, consumed_energy):
        """处理能耗报告"""
        current_time = self.bs.time()
        
        # 计算功率
        if self.last_energy_time > 0:  # 不是第一次更新
            time_diff = current_time - self.last_energy_time
            energy_diff = consumed_energy - self.last_energy
            
            if time_diff > 0:
                self.current_power = energy_diff / time_diff  # 计算功率（瓦特）
        
        self.last_energy = consumed_energy
        self.last_energy_time = current_time

    def onDeadlock(self):
        """处理死锁情况"""
        print("Deadlock detected at time:", self.bs.time())
        raise ValueError("Batsim has reached a deadlock")

    def onSimulationEnds(self):
        """模拟结束时的回调"""
        print("Simulation ends at time:", self.bs.time())
        super().onSimulationEnds()

    def try_schedule_jobs(self):
        """尝试调度等待队列中的作业（FCFS算法）"""
        if not self.waiting_jobs:
            return
            
        # 获取可用节点（只考虑ACTIVE和IDLE状态的节点）
        available_nodes = self.power_controller.GetAvailableNodes()
        if not available_nodes:
            return
            
        jobs_to_execute = []
        remaining_nodes = available_nodes.copy()
        
        i = 0
        while i < len(self.waiting_jobs):
            job = self.waiting_jobs[i]
            
            # 检查是否有足够的节点
            if job.requested_resources <= len(remaining_nodes):
                # 分配资源
                selected_nodes = remaining_nodes[:job.requested_resources]
                job.allocation = ProcSet(*selected_nodes)
                
                # 更新节点状态和作业数量
                for node in selected_nodes:
                    self.power_controller.AddJobToNode(node)
                
                # 从可用节点列表和等待队列中移除
                remaining_nodes = remaining_nodes[job.requested_resources:]
                self.waiting_jobs.pop(i)
                
                # 添加到待执行列表和运行队列
                jobs_to_execute.append(job)
                self.running_jobs.append(job)
            else:
                i += 1
                
        # 执行作业
        if jobs_to_execute:
            self.bs.execute_jobs(jobs_to_execute)
