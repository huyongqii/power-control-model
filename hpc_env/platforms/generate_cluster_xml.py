#!/usr/bin/env python3

"""
生成Batsim集群平台配置文件
使用XML生成器确保格式正确
"""

import argparse
from typing import Dict
import xml.etree.ElementTree as ET
from xml.dom import minidom
import yaml

import sys
import os

# 添加父目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(BASE_DIR,'hpc_env', 'data')
CONFIG_DIR = os.path.join(BASE_DIR,'hpc_env', 'platforms')

def load_config(config_file='config.yaml'):
    """加载配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class ClusterXMLGenerator:
    """集群XML配置文件生成器"""
    
    def __init__(self, config_file='config.yaml'):
        config = load_config(config_file)
        self.cluster_config = config['cluster']
        
    def generate(self, output_file: str):
        """生成XML配置文件"""
        # 创建根元素
        platform = ET.Element('platform')
        platform.set('version', '4.1')
        
        # 创建zone元素
        zone = ET.SubElement(platform, 'zone')
        zone.set('id', 'AS0')
        zone.set('routing', 'Full')
        
        # 创建master节点
        master = ET.SubElement(zone, 'host')
        master.set('id', 'master_host')
        master.set('speed', self.cluster_config['node_speed'])
        master.set('core', str(self.cluster_config['node_cores']))
        
        # master节点功耗属性
        master_wattage = ET.SubElement(master, 'prop')
        master_wattage.set('id', 'wattage_per_state')
        master_wattage.set('value', f"{self.cluster_config['power']['idle']}:{self.cluster_config['power']['full']}, "10:20")
        print(self.cluster_config['power'])
        master_wattage_off = ET.SubElement(master, 'prop')
        master_wattage_off.set('id', 'wattage_off')
        master_wattage_off.set('value', str(self.cluster_config['power']['off_']))
        
        # 创建计算节点
        for i in range(self.cluster_config['num_nodes']):
            node = ET.SubElement(zone, 'host')
            node.set('id', f'node{i}')
            node.set('speed', self.cluster_config['node_speed'])
            node.set('core', str(self.cluster_config['node_cores']))
            
            node_role = ET.SubElement(node, 'prop')
            node_role.set('id', 'role')
            node_role.set('value', 'compute_node')
            
            # 定义主机在不同电源状态下的功耗，格式为 state1:state2:...，单位通常为瓦特（W）。
            node_wattage = ET.SubElement(node, 'prop')
            node_wattage.set('id', 'wattage_per_state')
            node_wattage.set('value', f"{self.cluster_config['power']['idle']}:{self.cluster_config['power']['full']}")
            
            # 定义主机在关机状态下的功耗，单位通常为瓦特（W）。
            node_wattage_off = ET.SubElement(node, 'prop')
            node_wattage_off.set('id', 'wattage_off')
            node_wattage_off.set('value', str(self.cluster_config['power']['off_']))
        
        # 生成XML字符串并格式化
        xml_str = minidom.parseString(ET.tostring(platform)).toprettyxml(indent='    ')
        
        # 添加DOCTYPE声明
        final_xml = '<?xml version="1.0"?>\n<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">\n' + xml_str[xml_str.find('<platform'):]
        
        # 保存到文件
        with open(output_file, 'w') as f:
            f.write(final_xml)
            
        print(f"Generated cluster configuration with {self.cluster_config['num_nodes']} compute nodes: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Batsim cluster platform XML')
    parser.add_argument('--config', type=str, default=os.path.join(CONFIG_DIR, 'config.yaml'),
                      help='Configuration file path')
    parser.add_argument('--output', type=str, default=os.path.join(OUTPUT_DIR, 'cluster.xml'),
                      help='Output XML file path')
    
    args = parser.parse_args()
    
    generator = ClusterXMLGenerator(args.config)
    generator.generate(args.output)

if __name__ == "__main__":
    main() 