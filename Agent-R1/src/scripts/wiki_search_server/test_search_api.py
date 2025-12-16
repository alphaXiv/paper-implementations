import requests
import time
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import statistics
import os

# 禁用代理
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

def test_search(url, query, top_k=10):
    """测试单个搜索请求"""
    start_time = time.time()
    response = requests.get(f"{url}/search", params={"query": query, "top_k": top_k})
    end_time = time.time()
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    client_time = end_time - start_time
    
    return {
        "query": query,
        "client_time": client_time,
        "server_total_time": result["total_time"],
        "server_search_time": result["search_time"],
        "num_results": len(result["query_results"][0]["results"]),
        "top_score": result["query_results"][0]["results"][0]["score"] if result["query_results"][0]["results"] else None
    }

def test_batch_search(url, queries, top_k=10):
    """测试批量搜索请求"""
    start_time = time.time()
    response = requests.post(
        f"{url}/search", 
        json={"queries": queries, "top_k": top_k}
    )
    end_time = time.time()
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    result = response.json()
    client_time = end_time - start_time

    print(queries[0])
    print(result["query_results"][0]["results"])
    
    # 计算每个查询的平均结果数和平均得分
    avg_results = sum(len(qr["results"]) for qr in result["query_results"]) / len(result["query_results"])
    avg_top_score = sum(qr["results"][0]["score"] if qr["results"] else 0 for qr in result["query_results"]) / len(result["query_results"])
    
    return {
        "batch_size": len(queries),
        "client_time": client_time,
        "server_total_time": result["total_time"],
        "server_search_time": result["search_time"],
        "avg_results_per_query": avg_results,
        "avg_top_score": avg_top_score,
        "time_per_query": client_time / len(queries)
    }

def run_benchmark(url, queries, top_k=10, concurrency=1, runs=1, batch_size=1):
    """运行基准测试"""
    all_results = []
    batch_results = []
    
    # 重复运行指定次数
    for run in range(runs):
        print(f"Run {run+1}/{runs}")
        
        if batch_size > 1:
            # 批量查询测试
            print("\n--- 批量查询测试 ---")
            # 将查询分成批次
            for i in range(0, len(queries), batch_size):
                batch_queries = queries[i:i+batch_size]
                if len(batch_queries) < batch_size and i > 0:
                    # 如果最后一批不足batch_size，跳过
                    continue
                    
                print(f"测试批量查询 (批次大小: {len(batch_queries)})")
                result = test_batch_search(url, batch_queries, top_k)
                if result:
                    batch_results.append(result)
                    print(f"批次大小: {result['batch_size']}")
                    print(f"  客户端时间: {result['client_time']:.4f}s")
                    print(f"  服务器总时间: {result['server_total_time']:.4f}s")
                    print(f"  服务器搜索时间: {result['server_search_time']:.4f}s")
                    print(f"  每个查询平均时间: {result['time_per_query']:.4f}s")
                    print(f"  每个查询平均结果数: {result['avg_results_per_query']:.2f}")
        
        # 单个查询测试
        print("\n--- 单个查询测试 ---")
        # 创建查询列表
        query_list = []
        for query in queries:
            for _ in range(concurrency):
                query_list.append((url, query, top_k))
        
        # 使用线程池并发执行查询
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(test_search, *args) for args in query_list]
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"查询: {result['query']}")
                    print(f"  客户端时间: {result['client_time']:.4f}s")
                    print(f"  服务器总时间: {result['server_total_time']:.4f}s")
                    print(f"  服务器搜索时间: {result['server_search_time']:.4f}s")
                    print(f"  结果数: {result['num_results']}")
                    print(f"  最高得分: {result['top_score']}")
        
        all_results.extend(results)
    
    return all_results, batch_results

def analyze_results(results):
    """分析测试结果"""
    if not results:
        return
    
    # 提取时间数据
    client_times = [r["client_time"] for r in results]
    server_total_times = [r["server_total_time"] for r in results]
    server_search_times = [r["server_search_time"] for r in results]
    
    # 计算统计数据
    stats = {
        "client_time": {
            "min": min(client_times),
            "max": max(client_times),
            "mean": statistics.mean(client_times),
            "median": statistics.median(client_times),
            "p95": sorted(client_times)[int(len(client_times) * 0.95)],
            "p99": sorted(client_times)[int(len(client_times) * 0.99)] if len(client_times) >= 100 else None
        },
        "server_total_time": {
            "min": min(server_total_times),
            "max": max(server_total_times),
            "mean": statistics.mean(server_total_times),
            "median": statistics.median(server_total_times),
            "p95": sorted(server_total_times)[int(len(server_total_times) * 0.95)],
            "p99": sorted(server_total_times)[int(len(server_total_times) * 0.99)] if len(server_total_times) >= 100 else None
        },
        "server_search_time": {
            "min": min(server_search_times),
            "max": max(server_search_times),
            "mean": statistics.mean(server_search_times),
            "median": statistics.median(server_search_times),
            "p95": sorted(server_search_times)[int(len(server_search_times) * 0.95)],
            "p99": sorted(server_search_times)[int(len(server_search_times) * 0.99)] if len(server_search_times) >= 100 else None
        }
    }
    
    return stats

def analyze_batch_results(results):
    """分析批量查询测试结果"""
    if not results:
        return
    
    # 按批次大小分组
    batch_sizes = sorted(set(r["batch_size"] for r in results))
    grouped_results = {size: [] for size in batch_sizes}
    
    for result in results:
        grouped_results[result["batch_size"]].append(result)
    
    # 计算每个批次大小的统计数据
    stats = {}
    for size, size_results in grouped_results.items():
        client_times = [r["client_time"] for r in size_results]
        server_total_times = [r["server_total_time"] for r in size_results]
        server_search_times = [r["server_search_time"] for r in size_results]
        time_per_query = [r["time_per_query"] for r in size_results]
        
        stats[size] = {
            "client_time": {
                "min": min(client_times),
                "max": max(client_times),
                "mean": statistics.mean(client_times),
                "median": statistics.median(client_times)
            },
            "server_total_time": {
                "min": min(server_total_times),
                "max": max(server_total_times),
                "mean": statistics.mean(server_total_times),
                "median": statistics.median(server_total_times)
            },
            "server_search_time": {
                "min": min(server_search_times),
                "max": max(server_search_times),
                "mean": statistics.mean(server_search_times),
                "median": statistics.median(server_search_times)
            },
            "time_per_query": {
                "min": min(time_per_query),
                "max": max(time_per_query),
                "mean": statistics.mean(time_per_query),
                "median": statistics.median(time_per_query)
            }
        }
    
    return stats

def print_stats(stats):
    """打印统计数据"""
    print("\n===== 单个查询性能统计 =====")
    
    for metric, values in stats.items():
        print(f"\n{metric}:")
        for stat, value in values.items():
            if value is not None:
                print(f"  {stat}: {value:.4f}s")
    
    # 打印总结
    print("\n===== 总结 =====")
    print(f"平均客户端响应时间: {stats['client_time']['mean']:.4f}s")
    print(f"平均服务器总处理时间: {stats['server_total_time']['mean']:.4f}s")
    print(f"平均FAISS搜索时间: {stats['server_search_time']['mean']:.4f}s")
    print(f"嵌入生成时间 (估计): {stats['server_total_time']['mean'] - stats['server_search_time']['mean']:.4f}s")

def print_batch_stats(batch_stats):
    """打印批量查询统计数据"""
    if not batch_stats:
        return
    
    print("\n===== 批量查询性能统计 =====")
    
    for batch_size, stats in sorted(batch_stats.items()):
        print(f"\n批次大小: {batch_size}")
        print(f"  平均批次总时间: {stats['client_time']['mean']:.4f}s")
        print(f"  平均每个查询时间: {stats['time_per_query']['mean']:.4f}s")
        print(f"  加速比 (相对于单个查询): {1.0 / stats['time_per_query']['mean']:.2f}x")

def main():
    parser = argparse.ArgumentParser(description="测试搜索API性能")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="API URL")
    parser.add_argument("--queries", type=str, default="queries.txt", help="查询文件，每行一个查询")
    parser.add_argument("--top_k", type=int, default=10, help="每次搜索返回的结果数")
    parser.add_argument("--concurrency", type=int, default=1, help="并发查询数")
    parser.add_argument("--runs", type=int, default=1, help="重复运行次数")
    parser.add_argument("--batch_size", type=int, default=1, help="批量查询大小 (1表示不使用批量查询)")
    args = parser.parse_args()
    
    # 读取查询
    try:
        with open(args.queries, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"查询文件 {args.queries} 不存在，使用默认查询")
        queries = [
            "What is artificial intelligence?",
            "How does quantum computing work?",
            "Who was Albert Einstein?",
            "What is the history of the internet?",
            "Explain climate change"
        ]
    
    print(f"测试URL: {args.url}")
    print(f"查询数: {len(queries)}")
    print(f"并发数: {args.concurrency}")
    print(f"运行次数: {args.runs}")
    print(f"每次返回结果数: {args.top_k}")
    print(f"批量查询大小: {args.batch_size}")
    
    # 检查API是否可用
    try:
        health_response = requests.get(f"{args.url}/health")
        if health_response.status_code != 200:
            print(f"API健康检查失败: {health_response.status_code}")
            print(health_response.text)
            return
    except requests.exceptions.RequestException as e:
        print(f"无法连接到API: {e}")
        return
    
    # 运行测试
    results, batch_results = run_benchmark(
        args.url, queries, args.top_k, args.concurrency, args.runs, args.batch_size
    )
    
    # 分析结果
    stats = analyze_results(results)
    batch_stats = analyze_batch_results(batch_results)
    
    if stats:
        print_stats(stats)
    
    if batch_stats:
        print_batch_stats(batch_stats)
    
    # 保存结果
    with open("benchmark_results.json", "w") as f:
        json.dump({
            "config": vars(args),
            "results": results,
            "batch_results": batch_results,
            "stats": stats,
            "batch_stats": batch_stats
        }, f, indent=2)
    
    print(f"\n结果已保存到 benchmark_results.json")

if __name__ == "__main__":
    main() 