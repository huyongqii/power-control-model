from typing import Dict, List
import numpy as np
from datetime import datetime, timedelta

class PowerMetrics:
    """电源管理相关指标计算"""
    
    @staticmethod
    def calculate_power_metrics(cluster_states: List[Dict]) -> Dict:
        """计算电源相关指标"""
        if not cluster_states:
            return {}
            
        total_nodes = len(cluster_states[0]['nodes'])
        metrics = {
            'avg_active_nodes': 0,
            'avg_cpu_usage': 0,
            'avg_memory_usage': 0,
            'power_state_changes': 0,
            'avg_idle_time': 0
        }
        
        prev_state = None
        for state in cluster_states:
            # 计算活跃节点比例
            active_nodes = sum(1 for node in state['nodes'].values() 
                             if node['power_state'] == 'ON')
            metrics['avg_active_nodes'] += active_nodes / total_nodes
            
            # 计算资源使用率
            cpu_usage = sum(node['cpu_usage'] 
                          for node in state['nodes'].values()) / total_nodes
            memory_usage = sum(node['memory_usage'] 
                             for node in state['nodes'].values()) / total_nodes
            metrics['avg_cpu_usage'] += cpu_usage
            metrics['avg_memory_usage'] += memory_usage
            
            # 计算平均空闲时间
            idle_times = [node['idle_time'] 
                         for node in state['nodes'].values()]
            metrics['avg_idle_time'] += sum(idle_times) / total_nodes
            
            # 计算状态变化次数
            if prev_state:
                for node_id, node in state['nodes'].items():
                    if (node['power_state'] != 
                        prev_state['nodes'][node_id]['power_state']):
                        metrics['power_state_changes'] += 1
            
            prev_state = state
            
        # 计算平均值
        n_states = len(cluster_states)
        metrics['avg_active_nodes'] /= n_states
        metrics['avg_cpu_usage'] /= n_states
        metrics['avg_memory_usage'] /= n_states
        metrics['avg_idle_time'] /= n_states
        
        return metrics
        
    @staticmethod
    def calculate_prediction_accuracy(predictions: List[Dict],
                                   actual_states: List[Dict]) -> float:
        """计算预测准确率"""
        if not predictions or not actual_states:
            return 0.0
            
        errors = []
        for pred, actual in zip(predictions, actual_states):
            pred_nodes = pred['active_nodes']
            actual_nodes = sum(1 for node in actual['nodes'].values() 
                             if node['power_state'] == 'ON')
            error = abs(pred_nodes - actual_nodes) / len(actual['nodes'])
            errors.append(error)
            
        return 1 - np.mean(errors)  # 转换为准确率
        
    @staticmethod
    def calculate_energy_savings(baseline_states: List[Dict],
                               actual_states: List[Dict]) -> float:
        """计算节能效果"""
        if not baseline_states or not actual_states:
            return 0.0
            
        baseline_energy = sum(
            sum(1 for node in state['nodes'].values() 
                if node['power_state'] == 'ON')
            for state in baseline_states
        )
        
        actual_energy = sum(
            sum(1 for node in state['nodes'].values() 
                if node['power_state'] == 'ON')
            for state in actual_states
        )
        
        return 1 - (actual_energy / baseline_energy) 