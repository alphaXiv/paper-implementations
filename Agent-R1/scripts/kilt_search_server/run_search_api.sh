#!/bin/bash

# 设置环境变量
export INDEX_PATH="../../data/corpus/kilt/kilt_index_IVF16384_PQ64.bin"

# 安装依赖
# pip install -r requirements.txt

# 启动API服务
echo "启动搜索API服务..."
python search_api.py 