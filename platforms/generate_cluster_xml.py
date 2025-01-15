#!/usr/bin/env python3

"""
生成Batsim集群平台配置文件
使用XML生成器确保格式正确
"""

import argparse
from typing import Dict
import xml.etree.ElementTree as ET
from xml.dom import minidom

NUM_NODES = 500

class ClusterXMLGenerator:
    """集群XML配置文件生成器"""
    
    def __init__(self, config: Dict):
        self.num_nodes = config.get('num_nodes', NUM_NODES)
        
    def generate(self, output_file: str):
        """生成XML配置文件"""
        # 创建根元素
        platform = ET.Element('platform')
        platform.set('version', '4.1')
        
        # 创建zone元素
        zone = ET.SubElement(platform, 'zone')
        zone.set('id', 'AS0')
        zone.set('routing', 'Full')
        
        # 创建master节点 - 不显式设置role
        master = ET.SubElement(zone, 'host')
        master.set('id', 'master_host')
        master.set('speed', '40Gf')
        master.set('core', '56')
        
        # master节点功耗属性
        master_wattage = ET.SubElement(master, 'prop')
        master_wattage.set('id', 'wattage_per_state')
        master_wattage.set('value', '250.0:800.0')
        
        master_wattage_off = ET.SubElement(master, 'prop')
        master_wattage_off.set('id', 'wattage_off')
        master_wattage_off.set('value', '10')
        
        # 创建计算节点
        for i in range(self.num_nodes):
            node = ET.SubElement(zone, 'host')
            node.set('id', f'node{i}')
            # 主机的处理速度，通常以 GHz 或 GFLOPS（浮点运算每秒）表示
            node.set('speed', '40Gf')
            # 主机的核心数
            node.set('core', '56')
            
            node_role = ET.SubElement(node, 'prop')
            node_role.set('id', 'role')
            node_role.set('value', 'compute_node')
            
            # 定义主机在不同电源状态下的功耗，格式为 state1:state2:...，单位通常为瓦特（W）。
            node_wattage = ET.SubElement(node, 'prop')
            node_wattage.set('id', 'wattage_per_state')
            node_wattage.set('value', '250:800')
            
            # 定义主机在关机状态下的功耗，单位通常为瓦特（W）。
            node_wattage_off = ET.SubElement(node, 'prop')
            node_wattage_off.set('id', 'wattage_off')
            node_wattage_off.set('value', '10')
        
        # 生成XML字符串并格式化
        xml_str = minidom.parseString(ET.tostring(platform)).toprettyxml(indent='    ')
        
        # 添加DOCTYPE声明
        final_xml = '<?xml version="1.0"?>\n<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">\n' + xml_str[xml_str.find('<platform'):]
        
        # 保存到文件
        with open(output_file, 'w') as f:
            f.write(final_xml)
            
        print(f"Generated cluster configuration with {self.num_nodes} compute nodes: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Batsim cluster platform XML')
    parser.add_argument('--nodes', type=int, default=NUM_NODES,
                      help='Number of compute nodes')
    parser.add_argument('--output', type=str, default='data/cluster.xml',
                      help='Output XML file path')
    
    args = parser.parse_args()
    
    config = {
        'num_nodes': args.nodes
    }
    
    generator = ClusterXMLGenerator(config)
    generator.generate(args.output)

if __name__ == "__main__":
    main() 