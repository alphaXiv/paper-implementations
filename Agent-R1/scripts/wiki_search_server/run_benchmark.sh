#!/bin/bash

# 禁用代理
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

# 默认参数
URL="http://localhost:8000"
QUERIES="queries.txt"
TOP_K=5
CONCURRENCY=4
RUNS=3
BATCH_SIZE=64

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --url)
      URL="$2"
      shift 2
      ;;
    --queries)
      QUERIES="$2"
      shift 2
      ;;
    --top_k)
      TOP_K="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --runs)
      RUNS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

echo "运行基准测试..."
echo "URL: $URL"
echo "查询文件: $QUERIES"
echo "每次返回结果数: $TOP_K"
echo "并发数: $CONCURRENCY"
echo "运行次数: $RUNS"
echo "批量查询大小: $BATCH_SIZE"

# 运行测试
python test_search_api.py --url "$URL" --queries "$QUERIES" --top_k "$TOP_K" --concurrency "$CONCURRENCY" --runs "$RUNS" --batch_size "$BATCH_SIZE" 