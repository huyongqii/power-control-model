import os
from typing import List, Dict, Tuple
from enum import Enum
import requests
from datetime import datetime, timedelta

# 设置起始时间
start_time = datetime(2025, 1, 1)

class NodeState(Enum):
    ACTIVE = "active"          # 活跃状态
    IDLE = "idle"             # 空闲状态
    SLEEPING = "sleeping"      # 睡眠状态
    POWERED_OFF = "powered_off"  # 关机状态
    
    # 临界态
    SWITCHING_TO_SLEEP = "switching_to_sleep"    # 正在进入睡眠
    WAKING_FROM_SLEEP = "waking_from_sleep"      # 正在从睡眠唤醒
    POWERING_ON = "powering_on"                  # 正在开机
    POWERING_OFF = "powering_off"                # 正在关机

class PState(Enum):
    ON_NUM = 0 # 开机状态
    SLEEP_NUM = 1 # 睡眠状态
    SWITCHING_SLEEPING_NUM = 2 # 正在睡眠
    SWITCHING_WAKEUP_NUM = 3 # 正在唤醒

class NodePowerController:
    def __init__(self, 
                 batsim_scheduler,  # batsim调度器实例
                 predictor_url='http://localhost:5000',  # 预测服务的URL
                 min_uptime: int = 600,      # 最短开机保持时长(秒)
                 min_downtime: int = 300,    # 最短关机保持时长(秒)
                 sleep_threshold: int = 2160000,  # 空闲超过6小时转为关机
                 buffer_ratio: float = 0.1,  # 冗余比例
                 batch_size: int = 5):        # 每批次最大调整数量
        self.bs = batsim_scheduler
        self.predictor_url = predictor_url
        self.min_uptime = min_uptime
        self.min_downtime = min_downtime
        self.sleep_threshold = sleep_threshold
        self.buffer_ratio = buffer_ratio
        self.batch_size = batch_size
        # 添加功率相关的属性
        self.last_energy = 0.0        # 上次能耗值
        self.last_energy_time = 0.0   # 上次能耗时间
        self.current_power = 0.0      # 当前功率

        if os.path.exists('record.csv'):
        os.remove('record.csv')
        
        # 记录节点状态的字典
        self.node_states: Dict[str, Dict] = {}
        
        # 初始化所有节点为空闲状态
        for node in range(self.bs.nb_resources):
            self.node_states[node] = {
                'state': NodeState.IDLE,
                'last_state_change': self.bs.time(),
                'job_count': 0  # 记录节点上的作业数量
            }

        # 创建状态转换日志文件
        with open('node_state_changes.csv', 'w') as f:
            f.write("time,node_id,old_state,new_state\n")

    def AddJobToNode(self, node_id: str) -> bool:
        """
        节点添加一个作业
        返回: 是否添加成功
        """
        return self._update_node_job_count(node_id, 1)

    def RemoveJobFromNode(self, node_id: str) -> bool:
        """
        节点移除一个作业
        返回: 是否移除成功
        """
        return self._update_node_job_count(node_id, -1)

    def GetNodeJobCount(self, node_id: str) -> int:
        """获取节点当前的作业数量"""
        if node_id in self.node_states:
            return self.node_states[node_id]['job_count']
        return 0
    
    def GetActiveNodes(self) -> List[str]:
        """获取活跃状态的节点列表"""
        return [node_id for node_id, info in self.node_states.items() 
                if info['state'] == NodeState.ACTIVE]

    def GetIdleNodes(self) -> List[str]:
        """获取空闲状态的节点列表"""
        return [node_id for node_id, info in self.node_states.items() 
                if info['state'] == NodeState.IDLE]

    def GetSleepingNodes(self) -> List[str]:
        """获取睡眠状态的节点列表"""
        return [node_id for node_id, info in self.node_states.items() 
                if info['state'] == NodeState.SLEEPING]

    def GetPoweredOffNodes(self) -> List[str]:
        """获取关机状态的节点列表"""
        return [node_id for node_id, info in self.node_states.items() 
                if info['state'] == NodeState.POWERED_OFF]

    def ExecutePowerActions(self):
        """
        执行电源管理操作，基于当前系统状态自动进行电源管理
        """
        # 获取当前状态
        current_active = len(self.GetActiveNodes())
        idle_nodes = self.GetIdleNodes()
        sleeping_nodes = self.GetSleepingNodes()
        powered_off_nodes = self.GetPoweredOffNodes()
        
        # 获取预测的活跃节点数
        predicted_active = self._get_predicted_active_nodes()
        
        # 如果预测值为-1（数据不足），则不执行任何操作
        if predicted_active == -1:
            return 0, 0, 0, 0
        
        # 获取需要执行的电源操作
        nodes_to_wake, nodes_to_power_on, nodes_to_sleep, nodes_to_shutdown = \
            self._make_power_decision(predicted_active, current_active, 
                                   idle_nodes, sleeping_nodes, powered_off_nodes)
        
        # 执行唤醒操作
        for node in nodes_to_wake:
            self._wake_up_node(node)
            
        # 执行开机操作
        for node in nodes_to_power_on:
            self._switch_on_node(node)
            
        # 执行睡眠操作
        for node in nodes_to_sleep:
            self._sleep_node(node)
            
        # 执行关机操作
        for node in nodes_to_shutdown:
            self._switch_off_node(node)
            
        return len(nodes_to_wake), len(nodes_to_power_on), \
               len(nodes_to_sleep), len(nodes_to_shutdown)

    def GetAvailableNodes(self) -> List[str]:
        """获取可用于作业调度的节点列表（只返回完全活跃和空闲的节点）"""
        return [node_id for node_id, info in self.node_states.items() 
                if info['state'] in [NodeState.ACTIVE, NodeState.IDLE]]

    def GetNodeState(self, node_id: str) -> NodeState:
        """获取指定节点的状态"""
        if node_id in self.node_states:
            return self.node_states[node_id]['state']
        return None

    def GetNodeStateDuration(self, node_id: str) -> float:
        """获取节点在当前状态持续的时间（秒）"""
        if node_id in self.node_states:
            return self.bs.time() - self.node_states[node_id]['last_state_change']
        return 0

    def HandleStateTransitionComplete(self, node_id: any, new_power_state: any):
        """
        处理节点状态转换完成
        
        参数:
            node_id: 节点ID
            target_state: 转换完成后的目标状态
        """
        if node_id not in self.node_states:
            assert False, f"Node {node_id} not found in node_states"
        
        new_power_state_num = int(new_power_state)
        old_node_state = self.node_states[node_id]['state']
        if new_power_state_num == PState.ON_NUM.value:
            if old_node_state == NodeState.WAKING_FROM_SLEEP:
                self._update_node_state(node_id, NodeState.IDLE)
        elif new_power_state_num == PState.SLEEP_NUM.value:
            if old_node_state == NodeState.SWITCHING_TO_SLEEP:
                self._update_node_state(node_id, NodeState.SLEEPING)
        else:
            assert False, f"Invalid state transition from {old_node_state} to {new_power_state_num} for node {node_id}"
            
    def _switch_on_node(self, node_id: str):
        """开启指定节点"""
        assert "Do not use this function"
        self.bs.set_resource_state(f"{node_id}-{node_id}", str(PState.ON_NUM.value))  # 使用正确的格式和状态值
        self._update_node_state(node_id, NodeState.POWERING_ON)  # 先设置为开机中状态
    
    def _switch_off_node(self, node_id: str):
        """关闭指定节点"""
        assert "Do not use this function"
        self.bs.set_resource_state(f"{node_id}-{node_id}", str(PState.SLEEP_NUM.value))  # 使用正确的格式和状态值
        self._update_node_state(node_id, NodeState.POWERING_OFF)  # 先设置为关机中状态
    
    def _sleep_node(self, node_id: str):
        """使节点进入睡眠状态"""
        self.bs.set_resource_state(f"{node_id}-{node_id}", str(PState.SLEEP_NUM.value))  # 使用正确的格式和状态值
        self._update_node_state(node_id, NodeState.SWITCHING_TO_SLEEP)  # 先设置为睡眠中状态
    
    def _wake_up_node(self, node_id: str):
        """唤醒睡眠节点"""
        self.bs.set_resource_state(f"{node_id}-{node_id}", str(PState.ON_NUM.value))  # 使用正确的格式和状态值
        self._update_node_state(node_id, NodeState.WAKING_FROM_SLEEP)  # 先设置为唤醒中状态
    
    def _update_node_state(self, node_id: str, state: NodeState):
        """内部方法：更新节点状态"""
        current_time = self.bs.time()
        
        if node_id not in self.node_states:
            self.node_states[node_id] = {
                'state': state,
                'last_state_change': current_time,
                'job_count': 0
            }
            # 记录新节点的初始状态
            self._record_state_change(current_time, node_id, None, state)
        elif self.node_states[node_id]['state'] != state:
            old_state = self.node_states[node_id]['state']
            self.node_states[node_id].update({
                'state': state,
                'last_state_change': current_time
            })
            # 记录状态变化
            self._record_state_change(current_time, node_id, old_state, state)

    def _record_state_change(self, time: float, node_id: str, old_state: NodeState, new_state: NodeState):
        """记录节点状态变化
        
        参数:
            time: 发生变化的时间
            node_id: 节点ID
            old_state: 原状态（对于新节点可能为None）
            new_state: 新状态
        """
        with open('node_state_changes.csv', 'a') as f:
            old_state_str = str(old_state.value) if old_state else "INIT"
            f.write(f"{time},{node_id},{old_state_str},{new_state.value}\n")

    def _update_node_job_count(self, node_id: str, job_change: int) -> bool:
        """
        更新节点的作业数量并相应更新节点状态
        
        参数:
            node_id: 节点ID
            job_change: 作业数量变化 (+1表示添加作业，-1表示移除作业)
        返回:
            bool: 更新是否成功
        """
        if node_id not in self.node_states:
            return False
        
        current_state = self.node_states[node_id]['state']
        # 只允许更新ACTIVE或IDLE状态的节点
        if current_state not in [NodeState.ACTIVE, NodeState.IDLE]:
            return False
        
        self.node_states[node_id]['job_count'] += job_change
        new_count = self.node_states[node_id]['job_count']
        
        # 防止作业数量为负
        if new_count < 0:
            self.node_states[node_id]['job_count'] = 0
            new_count = 0
        
        # 根据作业数量更新节点状态
        if new_count > 0:
            self._update_node_state(node_id, NodeState.ACTIVE)
        else:
            self._update_node_state(node_id, NodeState.IDLE)
        
        return True
    
    def _get_predicted_active_nodes(self) -> int:
        """
        从预测服务获取预测的未来活跃节点数量
        如果系统运行时间不足4小时，返回-1
        """
        # 检查系统运行时间是否达到4小时
        if self.bs.time() < 15000:  # 4小时 = 14400秒
            print(f"系统运行时间（{self.bs.time():.2f}秒）不足4小时，暂不执行预测")
            return -1
            
        try:
            response = requests.post(
                f"{self.predictor_url}/predict",
                json={'csv_path': '/root/PredictModel/pybatsim-3.2.1/schedulers/green-energy/power_control/record.csv'}
            )
            if response.status_code == 200:
                return response.json()['prediction']
            else:
                assert False, f"Prediction request failed: {response.json().get('error')}"
        except Exception as e:
            assert False, f"Failed to get prediction: {str(e)}"

    def _get_nodes_to_wake_up(self, 
                            predicted_active: int,
                            current_active: int,
                            sleeping_nodes: List[str],
                            powered_off_nodes: List[str]) -> Tuple[List[str], List[str]]:
        """决定需要唤醒的节点（包括从睡眠唤醒和从关机启动）"""
        if predicted_active <= current_active:
            return [], []
            
        # 计算需要额外的节点数量（包含冗余）
        needed = int((predicted_active - current_active) * (1 + self.buffer_ratio))
        needed = min(needed, self.batch_size)  # 应用批次限制
        
        # 优先从睡眠节点中唤醒
        current_time = self.bs.time()
        nodes_to_wake = []
        nodes_to_power_on = []
        
        # 先尝试唤醒睡眠节点
        if needed > 0:
            nodes_to_wake = sleeping_nodes[:needed]
            needed -= len(nodes_to_wake)
        
        # 如果睡眠节点不够，再从关机节点中选择
        if needed > 0:
            eligible_nodes = [
                node for node in powered_off_nodes
                if current_time - self.node_states[node]['last_state_change'] >= self.min_downtime
            ]
            nodes_to_power_on = eligible_nodes[:needed]
        
        return nodes_to_wake, nodes_to_power_on
    
    def _get_nodes_to_sleep_or_shutdown(self,
                                     predicted_active: int,
                                     current_active: int,
                                     idle_nodes: List[str]) -> Tuple[List[str], List[str]]:
        """决定需要睡眠或关机的节点"""
        if predicted_active >= current_active:
            return [], []
            
        # 计算需要减少的数量（保留冗余）
        excess = current_active - predicted_active
        to_reduce = int(excess * (1 - self.buffer_ratio))
        # to_reduce = min(to_reduce, self.batch_size)  # 应用批次限制
        
        current_time = self.bs.time()
        nodes_to_sleep = []
        nodes_to_shutdown = []
        
        # 检查每个空闲节点的状态
        for node in idle_nodes[:to_reduce]:
            if node not in self.node_states:
                nodes_to_sleep.append(node)
                continue
                
            idle_time = current_time - self.node_states[node]['last_state_change']
            
            # 如果空闲时间超过阈值，则关机
            if idle_time >= self.sleep_threshold:
                nodes_to_shutdown.append(node)
            # 否则进入睡眠状态
            else:
                nodes_to_sleep.append(node)
        
        return nodes_to_sleep, nodes_to_shutdown
    
    def _make_power_decision(self,
                           predicted_active: int,
                           current_active: int,
                           idle_nodes: List[str],
                           sleeping_nodes: List[str],
                           powered_off_nodes: List[str]) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        根据预测值和当前状态做出节点状态转换决策
        
        返回值:
            Tuple[List[str], List[str], List[str], List[str]]: 
            (要唤醒的睡眠节点列表, 要开机的关机节点列表, 要睡眠的节点列表, 要关机的节点列表)
        """
        # 处理需要增加活跃节点的情况
        nodes_to_wake, nodes_to_power_on = self._get_nodes_to_wake_up(
            predicted_active, current_active, sleeping_nodes, powered_off_nodes)
        
        # 处理需要减少活跃节点的情况
        nodes_to_sleep, nodes_to_shutdown = self._get_nodes_to_sleep_or_shutdown(
            predicted_active, current_active, idle_nodes)
            
        return nodes_to_wake, nodes_to_power_on, nodes_to_sleep, nodes_to_shutdown

    def RecordSystemState(self, current_time: float, running_jobs: list, waiting_jobs: list, current_power: float):
        """
        记录系统当前状态到CSV文件
        
        参数:
            current_time: 当前时间（秒）
            running_jobs: 正在运行的作业列表
            waiting_jobs: 等待中的作业列表
            current_power: 当前功率
        """
        nb_computing = len(self.GetActiveNodes())
        nb_idle = len(self.GetIdleNodes())
        nb_sleeping = len(self.GetSleepingNodes())
        nb_powered_off = len(self.GetPoweredOffNodes())
        
        # 获取临界态节点数量
        nb_switching_to_sleep = len([node_id for node_id, info in self.node_states.items() 
                                   if info['state'] == NodeState.SWITCHING_TO_SLEEP])
        nb_waking_from_sleep = len([node_id for node_id, info in self.node_states.items() 
                                   if info['state'] == NodeState.WAKING_FROM_SLEEP])
        nb_powering_on = len([node_id for node_id, info in self.node_states.items() 
                             if info['state'] == NodeState.POWERING_ON])
        nb_powering_off = len([node_id for node_id, info in self.node_states.items() 
                              if info['state'] == NodeState.POWERING_OFF])
        
        total_nodes = self.bs.nb_resources
        utilization_rate = nb_computing / total_nodes if total_nodes > 0 else 0
        
        # 将时间戳转换为datetime对象
        datetime_obj = start_time + timedelta(seconds=float(current_time))
        
        # 如果文件不存在，创建并写入表头
        if not os.path.exists('record.csv'):
            with open('record.csv', 'w') as f:
                f.write("timestamp,datetime,running_jobs,waiting_jobs,nb_computing,"
                        "utilization_rate,epower,nb_idle,nb_sleeping,nb_powered_off,"
                        "nb_switching_to_sleep,nb_waking_from_sleep,nb_powering_on,"
                        "nb_powering_off\n")
        
        # 写入当前状态
        with open('record.csv', 'a') as f:
            f.write(f"{current_time},{datetime_obj},{len(running_jobs)},"
                    f"{len(waiting_jobs)},{nb_computing},{utilization_rate:.4f},"
                    f"{current_power},{nb_idle},{nb_sleeping},{nb_powered_off},"
                    f"{nb_switching_to_sleep},{nb_waking_from_sleep},{nb_powering_on},"
                    f"{nb_powering_off}\n")
