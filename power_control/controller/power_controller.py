from typing import List, Dict
import numpy as np
from datetime import datetime, timedelta

class PowerController:
    """节点电源控制器"""
    
    def __init__(self, config: dict):
        self.config = config
        self.predictor = None
        self.last_operation_time = {}  # 记录节点最后操作时间
        
    def get_power_operations(self, 
                           cluster_state: Dict,
                           prediction: np.ndarray) -> List[Dict]:
        """
        根据集群状态和预测结果生成开关机操作
        
        Args:
            cluster_state: {
                'nodes': {
                    'node_id': {
                        'power_state': 'ON/OFF',
                        'cpu_usage': float,
                        'memory_usage': float,
                        'idle_time': float
                    }
                },
                'timestamp': datetime
            }
            prediction: 未来活跃节点数预测
        
        Returns:
            [{'node_id': str, 'operation': 'ON/OFF', 'reason': str}]
        """
        current_time = cluster_state['timestamp']
        required_nodes = self._calculate_required_nodes(prediction)
        
        # 当前活跃节点数
        active_nodes = sum(1 for node in cluster_state['nodes'].values() 
                         if node['power_state'] == 'ON')
        
        operations = []
        if required_nodes > active_nodes:
            # 需要开启更多节点
            nodes_to_on = self._select_nodes_to_power_on(
                cluster_state,
                required_nodes - active_nodes
            )
            operations.extend([
                {
                    'node_id': node_id,
                    'operation': 'ON',
                    'reason': 'Predicted load increase'
                }
                for node_id in nodes_to_on
            ])
        elif required_nodes < active_nodes:
            # 需要关闭一些节点
            nodes_to_off = self._select_nodes_to_power_off(
                cluster_state,
                active_nodes - required_nodes
            )
            operations.extend([
                {
                    'node_id': node_id,
                    'operation': 'OFF',
                    'reason': 'Low utilization predicted'
                }
                for node_id in nodes_to_off
            ])
            
        return operations
        
    def _calculate_required_nodes(self, prediction: np.ndarray) -> int:
        """计算所需节点数"""
        # 使用预测的最大值作为基准
        base_required = int(np.ceil(np.max(prediction)))
        
        # 添加安全边际
        safety_margin = int(base_required * self.config['safety_margin'])
        
        return base_required + safety_margin
        
    def _select_nodes_to_power_on(self, 
                                 cluster_state: Dict,
                                 count: int) -> List[str]:
        """选择要开启的节点"""
        # 获取所有关机节点
        off_nodes = [
            node_id for node_id, node in cluster_state['nodes'].items()
            if node['power_state'] == 'OFF'
        ]
        
        # 按最后操作时间排序（优先开启最近关闭的节点）
        sorted_nodes = sorted(
            off_nodes,
            key=lambda x: self.last_operation_time.get(x, datetime.min)
        )
        
        return sorted_nodes[:count]
        
    def _select_nodes_to_power_off(self, 
                                  cluster_state: Dict,
                                  count: int) -> List[str]:
        """选择要关闭的节点"""
        # 获取所有空闲节点
        idle_nodes = [
            node_id for node_id, node in cluster_state['nodes'].items()
            if (node['power_state'] == 'ON' and
                node['cpu_usage'] < self.config['idle_threshold'] and
                node['memory_usage'] < self.config['idle_threshold'] and
                node['idle_time'] > self.config['min_idle_time'])
        ]
        
        # 按空闲时间排序
        sorted_nodes = sorted(
            idle_nodes,
            key=lambda x: cluster_state['nodes'][x]['idle_time'],
            reverse=True
        )
        
        return sorted_nodes[:count] 