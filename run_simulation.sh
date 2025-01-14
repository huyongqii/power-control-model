#!/bin/bash

# 生成工作负载
python workloads/generate_jobs.py

# 启动BATSIM（带能耗监控）
batsim --platform platforms/cluster.xml \
       --workload workloads/jobs.json \
       --energy \
       --export result/out \
       --verbosity information

# 在另一个终端中运行调度器
# python schedulers/basic_scheduler.py
