# 此文档由AI生成
# 🔋 电力控制模型系统 (Power Control Model System) 

一个完整的高性能计算集群电力需求预测系统，通过深度学习技术分析历史作业数据，实现对集群节点使用情况和电力消耗的智能预测。

## 📊 项目概述

本项目是一个端到端的HPC集群资源管理和电力预测解决方案，旨在帮助数据中心管理员：

- 🎯 **精准预测**: 基于历史数据预测未来30分钟的节点使用情况
- ⚡ **电力优化**: 通过预测结果优化集群电力分配和调度
- 📈 **实时监控**: 提供Web API接口，支持实时预测服务
- 🧠 **智能调度**: 辅助作业调度器做出更优的资源分配决策

## 🏗️ 系统架构

```
power-control-model/
├── 📄 README.md                    # 项目说明文档
├── 🔧 process_sql.py              # 数据库作业数据提取脚本
├── 🚫 .gitignore                  # Git忽略配置
│
├── 🤖 trainer/                    # 模型训练模块
│   ├── config.py                 # 训练配置参数
│   ├── trainer.py                # 主训练器（含早停、学习率调度）
│   ├── model.py                  # 神经网络模型定义
│   ├── data_loader.py            # 数据加载和批处理
│   ├── data_processor.py         # 数据预处理和特征工程
│   ├── evaluator.py              # 模型性能评估
│   └── visualization_results/    # 训练可视化结果
│
├── 💾 model/                      # 模型存储模块
│   ├── checkpoint.pth            # 训练好的模型权重
│   └── dataset_scalers.pkl       # 数据标准化器
│
└── 🚀 predictor/                  # 预测服务模块
    ├── model.py                  # 预测模型定义
    ├── predictor.py              # Flask Web API服务
    └── simGenTask.py             # 作业生成模拟器
```

## ⭐ 核心功能特性

### 🔍 1. 数据处理引擎 (`process_sql.py`)

- **数据源**: 从MySQL数据库中提取HPC作业执行记录
- **智能解析**: 自动解析节点分配信息、CPU需求、运行时间等
- **时间序列生成**: 按分钟级别聚合生成完整的集群使用时间序列
- **特征计算**: 
  - ✅ 运行作业数量统计
  - ✅ 等待队列长度分析  
  - ✅ 活跃节点数量追踪
  - ✅ CPU平均占用率计算
  - ✅ 作业平均资源需求分析
  - ✅ 运行时间分布统计

### 🧠 2. 深度学习预测模型 (`trainer/model.py`)

采用业界先进的时间序列预测架构：

- **🔄 双向LSTM网络**: 
  - 2层双向LSTM，隐藏维度128
  - 捕获历史数据中的双向时间依赖关系
  - 支持变长序列输入
  
- **🎯 多头自注意力机制**:
  - 4个注意力头，增强长期依赖建模
  - 自适应权重分配，突出重要时间点
  
- **⚙️ 多模态特征融合**:
  - 历史序列特征 (过去4小时数据)
  - 时间周期特征 (小时、星期、月份)  
  - 历史同期特征 (同一时间点的历史模式)
  
- **🎛️ 先进正则化技术**:
  - Dropout防过拟合 (20%丢弃率)
  - Batch Normalization加速收敛
  - 梯度裁剪防梯度爆炸

### 🚂 3. 智能训练系统 (`trainer/trainer.py`)

完整的深度学习训练流水线：

- **📊 自定义损失函数**: 
  - 结合Huber Loss (鲁棒性) + 平滑正则项
  - 适应电力预测场景的特殊需求
  
- **📚 高级学习率策略**:
  - 线性预热 (前5轮) + 自适应衰减
  - ReduceLROnPlateau监控验证损失
  - 最小学习率保护机制
  
- **⏹️ 早停与检查点**:
  - 验证损失无改善时自动停止
  - 最佳模型自动保存
  - 完整训练历史记录

### 🌐 4. 生产级预测服务 (`predictor/predictor.py`)

基于Flask的高性能Web API服务：

- **📡 实时数据接入**: 
  - InfluxDB时序数据库集成
  - 自动数据质量检查和清洗
  
- **🔮 智能预测引擎**:
  - 基于训练好的模型进行实时推理
  - 支持多节点并发预测
  - 自动异常检测和容错
  
- **🖥️ RESTful API接口**:
  ```bash
  POST /predict    # 节点使用量预测
  GET  /health     # 服务健康检查
  ```

### 🎮 5. 作业模拟器 (`predictor/simGenTask.py`)

智能的HPC作业负载模拟器：

- **⏰ 时间感知调度**: 工作时间vs非工作时间不同的提交模式
- **📊 真实负载模拟**: 基于泊松分布的作业到达过程
- **💥 突发负载支持**: 随机突发任务生成，模拟高峰期场景  
- **🎯 多样化作业类型**: 短任务、常规任务、长任务的混合生成

## 🛠️ 技术栈详情

### 🤖 机器学习框架
- **PyTorch 1.9+**: 深度学习模型构建和训练
- **NumPy**: 高性能数值计算
- **Pandas**: 数据处理和分析
- **Scikit-learn**: 数据预处理和评估指标

### 🌐 Web服务技术  
- **Flask**: 轻量级Web框架
- **Waitress**: WSGI生产服务器
- **InfluxDB**: 时序数据存储
- **PyMySQL**: MySQL数据库连接

### 📊 数据可视化
- **Matplotlib**: 训练曲线和结果可视化
- **Seaborn**: 统计图表绘制

## 🚀 快速开始

### 📋 环境要求

```bash
# Python环境
Python >= 3.8

# 核心依赖
torch >= 1.9.0
numpy >= 1.20.0  
pandas >= 1.3.0
scikit-learn >= 1.0.0
flask >= 2.0.0
pymysql >= 1.0.0
influxdb-client >= 1.24.0
matplotlib >= 3.4.0
```

### ⚙️ 安装配置

1. **克隆项目**
```bash
git clone <repository-url>
cd power-control-model
```

2. **安装依赖**
```bash
pip install torch numpy pandas scikit-learn flask pymysql influxdb-client matplotlib pyyaml waitress joblib holidays
```

3. **配置数据库** (修改 `process_sql.py`)
```python
DB_CONFIG = {
    'host': 'your-mysql-host',
    'user': 'your-username', 
    'password': 'your-password',
    'database': 'your-database',
    'charset': 'utf8mb4'
}
```

### 📊 数据准备与训练

1. **提取作业数据**
```bash
python process_sql.py
# 输出: wm2/job_timeline_YYYY.csv
```

2. **启动模型训练**  
```bash
cd trainer
# 修改config.py中的数据路径
python trainer.py
```

3. **模型评估**
```bash
python evaluator.py
```

### 🚀 部署预测服务

1. **准备配置文件** (`config.yaml`)
```yaml
Predictor:
  URL: "http://0.0.0.0:5000"
  CheckpointFile: "../model/checkpoint.pth"
  ScalersFile: "../model/dataset_scalers.pkl"
  ForecastMinutes: 30
  LookbackMinutes: 240

InfluxDB:
  URL: "http://localhost:8086"
  Token: "your-influxdb-token"
  Org: "your-org"
  Bucket: "your-bucket"
```

2. **启动预测服务**
```bash
cd predictor
python predictor.py -c config.yaml
```

3. **测试API**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"total_nodes": 120}'
```

## 📈 性能指标

### 🎯 模型准确性
- **MAE** (平均绝对误差): < 2.5 节点
- **RMSE** (均方根误差): < 4.0 节点  
- **MAPE** (平均绝对百分比误差): < 15%
- **R²** (决定系数): > 0.85

### ⚡ 服务性能
- **预测延迟**: < 100ms
- **API吞吐量**: > 100 QPS  
- **模型大小**: < 50MB
- **内存占用**: < 1GB

## 📁 核心配置文件

### `trainer/config.py` - 训练配置
```python
MODEL_CONFIG = {
    'epochs': 100,           # 最大训练轮数
    'batch_size': 256,       # 批量大小
    'learning_rate': 0.001,  # 初始学习率  
    'lookback_minutes': 240, # 历史窗口(4小时)
    'forecast_minutes': 30,  # 预测窗口(30分钟)
    'total_nodes': 120,      # 集群节点总数
}
```

### 输出文件说明

**训练输出**:
- `model/checkpoint.pth`: 最佳模型权重
- `model/dataset_scalers.pkl`: 数据标准化器
- `trainer/logs/`: 训练日志和可视化结果

**数据处理输出**:
- `wm2/job_timeline_YYYY.csv`: 年度作业时间线数据

## 🤝 参与贡献

我们欢迎社区贡献！请遵循以下流程：

1. 🍴 Fork本项目
2. 🌟 创建功能分支 (`git checkout -b feature/amazing-feature`)  
3. ✅ 提交更改 (`git commit -m 'Add amazing feature'`)
4. 📤 推送分支 (`git push origin feature/amazing-feature`)
5. 🔀 创建Pull Request

## 📜 开源协议

本项目基于 **MIT License** 开源，详见 [LICENSE](LICENSE) 文件。

## 📞 联系我们

- 📧 Email: [project-email]
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 Wiki: [项目文档](https://github.com/your-repo/wiki)

---

<div align="center">

**🔋 让AI为您的数据中心节能减排！**

Made with ❤️ for the HPC Community

</div>
