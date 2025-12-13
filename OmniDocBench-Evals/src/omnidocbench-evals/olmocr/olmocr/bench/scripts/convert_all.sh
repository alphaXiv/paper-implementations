#!/bin/bash

# Exit on error but allow the trap to execute
set -e

# Global variables
SERVER_PID=""
BENCH_DIR=""  # New variable to store the benchmark directory

# Function to show usage information
show_usage() {
    echo "Usage: $0 [--dir <benchmark_directory>]"
    echo "  --dir   Specify the benchmark data directory (default: olmOCR-bench/bench_data/)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dir)
            BENCH_DIR="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

# Set default directory if not provided
if [ -z "$BENCH_DIR" ]; then
    BENCH_DIR="olmOCR-bench/bench_data/"
    echo "[INFO] Using default benchmark directory: $BENCH_DIR"
else
    echo "[INFO] Using benchmark directory: $BENCH_DIR"
fi

# Trap function to handle Ctrl+C (SIGINT)
cleanup() {
    echo -e "\n[INFO] Received interrupt signal. Cleaning up..."
    
    # Find and kill any Python processes started by this script
    echo "[INFO] Stopping any running Python processes"
    pkill -P $$ python || true
    
    # Stop server if running
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[INFO] Stopping server (PID: $SERVER_PID)"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi

    pkill vllm
    
    echo "[INFO] Cleanup complete. Exiting."
    exit 1
}

# Set the trap for SIGINT (Ctrl+C)
trap cleanup SIGINT

# Function to check if port 30000 is in use
check_port() {
    port=30000
    echo "[INFO] Checking if port $port is available..."
    
    if command -v lsof >/dev/null 2>&1; then
        # Linux/macOS
        if lsof -i :$port >/dev/null 2>&1; then
            echo "[ERROR] Port $port is already in use. Process details:"
            lsof -i :$port
            echo "[ERROR] Please stop the process using this port and try again."
            echo "        You can use: kill -9 <PID>"
            return 1
        fi
    elif command -v netstat >/dev/null 2>&1; then
        # Windows/other systems with netstat
        if netstat -an | grep -q ":$port "; then
            echo "[ERROR] Port $port is already in use. Process details:"
            if command -v findstr >/dev/null 2>&1; then
                # Windows
                netstat -ano | findstr ":$port"
                echo "[ERROR] Please stop the process using this port and try again."
                echo "        You can use: taskkill /F /PID <PID>"
            else
                netstat -an | grep ":$port "
                echo "[ERROR] Please stop the process using this port and try again."
            fi
            return 1
        fi
    else
        # Fallback method using nc if available
        if command -v nc >/dev/null 2>&1; then
            nc -z localhost $port >/dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo "[ERROR] Port $port is already in use, but cannot determine which process."
                echo "[ERROR] Please ensure port $port is available before continuing."
                return 1
            fi
        else
            echo "[WARNING] Cannot check if port $port is in use (neither lsof, netstat, nor nc available)."
            echo "[WARNING] Continuing anyway, but this might fail if the port is already in use."
            return 0
        fi
    fi
    
    echo "[INFO] Port $port is available."
    return 0
}

# Function to create conda environment if it doesn't exist
create_conda_env() {
    env_name=$1
    python_version=$2
    
    # Check if environment exists
    if conda info --envs | grep -q "^$env_name "; then
        echo "Environment $env_name already exists, using it."
    else
        echo "Creating conda environment: $env_name"
        conda create -y -n $env_name python=$python_version
    fi
}

# Generic function to start a server (either vllm or sglang)
start_server() {
    server_type=$1
    model_name=$2
    shift 2  # Remove server type and model name from the argument list
    
    echo "Starting $server_type server for model: $model_name"
    echo "Additional arguments: $@"
    
    if [ "$server_type" = "sglang" ]; then
        python -m sglang.launch_server --port 30000 --model "$model_name" "$@" &
    elif [ "$server_type" = "vllm" ]; then
        vllm serve $model_name --port 30000 "$@" &
    else
        echo "Unsupported server type: $server_type"
        exit 1
    fi
    SERVER_PID=$!
    
    # Check if the server process is running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Failed to start server process. Exiting."
        exit 1
    fi
    
    # Wait for the server to be ready by checking the models endpoint
    echo "Waiting for server to be ready..."
    max_attempts=300
    attempt=0
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:30000/v1/models" -o /dev/null -w "%{http_code}" | grep -q "200"; then
            echo "Server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        echo "Waiting for server... attempt $attempt/$max_attempts"
        sleep 2
    done
    
    echo "Server failed to become ready after multiple attempts. Exiting."
    kill $SERVER_PID
    SERVER_PID=""
    exit 1
}

# Function to stop the server
stop_server() {
    echo "Stopping server with PID: $SERVER_PID"
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null || true
        echo "Server stopped."
    else
        echo "No server to stop."
    fi
    SERVER_PID=""
}

# Create and activate olmocr environment
create_conda_env "olmocr" "3.11"
source $(conda info --base)/etc/profile.d/conda.sh
source activate olmocr
pip install -e .[bench]

# Run olmocr benchmarks, exactly as the pipeline.py does it
echo "Running olmocr benchmarks..."
python -m olmocr.bench.convert olmocr_pipeline --parallel 50 --dir "$BENCH_DIR"

# Install marker-pdf and run benchmarks
echo "Installing marker-pdf and running benchmarks..."
pip install marker-pdf==1.6.1
python -m olmocr.bench.convert marker --dir "$BENCH_DIR"

# Install verovio and run benchmarks
# echo "Installing verovio and running benchmarks..."
# pip install verovio
# python -m olmocr.bench.convert gotocr

# Run chatgpt benchmarks
echo "Running chatgpt benchmarks..."
python -m olmocr.bench.convert chatgpt --dir "$BENCH_DIR" --parallel 4
#python -m olmocr.bench.convert chatgpt:name=chatgpt45:model=gpt-4.5-preview-2025-02-27

# Run gemini benchmarks
echo "Running gemini benchmarks..."
python -m olmocr.bench.convert gemini:name=gemini_flash2:model=gemini-2.0-flash --parallel 4 --dir "$BENCH_DIR"

echo "Running mistral..."
pip install mistralai
python -m olmocr.bench.convert --dir "$BENCH_DIR" --parallel 4 mistral 

# Run raw server benchmarks with generic server function
# For each model, start server, run benchmark, then stop server

# Check port availability at script start
check_port || exit 1

# olmocr_base_temp0_1 using sglang server
# start_server sglang "allenai/olmOCR-7B-0225-preview" --chat-template qwen2-vl --mem-fraction-static 0.7
# python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=olmocr_base_temp0_0:model=allenai/olmOCR-7B-0225-preview:temperature=0.0:prompt_template=fine_tune:response_template=json --repeats 1 --parallel 50
# python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=olmocr_base_temp0_1:model=allenai/olmOCR-7B-0225-preview:temperature=0.1:prompt_template=fine_tune:response_template=json --repeats 5 --parallel 50
# python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=olmocr_base_temp0_2:model=allenai/olmOCR-7B-0225-preview:temperature=0.2:prompt_template=fine_tune:response_template=json --repeats 1 --parallel 50
# python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=olmocr_base_temp0_3:model=allenai/olmOCR-7B-0225-preview:temperature=0.3:prompt_template=fine_tune:response_template=json --repeats 1 --parallel 50
# python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=olmocr_base_temp0_4:model=allenai/olmOCR-7B-0225-preview:temperature=0.4:prompt_template=fine_tune:response_template=json --repeats 1 --parallel 50
# python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=olmocr_base_temp0_5:model=allenai/olmOCR-7B-0225-preview:temperature=0.5:prompt_template=fine_tune:response_template=json --repeats 1 --parallel 50
# python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=olmocr_base_temp0_6:model=allenai/olmOCR-7B-0225-preview:temperature=0.6:prompt_template=fine_tune:response_template=json --repeats 1 --parallel 50
# python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=olmocr_base_temp0_7:model=allenai/olmOCR-7B-0225-preview:temperature=0.7:prompt_template=fine_tune:response_template=json --repeats 5 --parallel 50

# python -m olmocr.bench.convert server:name=olmocr_base_temp0_1:model=allenai/olmOCR-7B-0225-preview:temperature=0.1:prompt_template=fine_tune:response_template=json --repeats 5 --parallel 50
# python -m olmocr.bench.convert server:name=olmocr_base_temp0_8:model=allenai/olmOCR-7B-0225-preview:temperature=0.8:prompt_template=fine_tune:response_template=json --repeats 5 --parallel 50
# stop_server

# start_server vllm "allenai/olmOCR-7B-0225-preview"
# python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=olmocr_base_temp_vllm0_1:model=allenai/olmOCR-7B-0225-preview:temperature=0.1:prompt_template=fine_tune:response_template=json --repeats 5 --parallel 50
# python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=olmocr_base_temp_vllm0_7:model=allenai/olmOCR-7B-0225-preview:temperature=0.7:prompt_template=fine_tune:response_template=json --repeats 5 --parallel 50

# python -m olmocr.bench.convert server:name=olmocr_base_vllm_temp0_1:model=allenai/olmOCR-7B-0225-preview:temperature=0.1:prompt_template=fine_tune:response_template=json --repeats 5 --parallel 50
# python -m olmocr.bench.convert server:name=olmocr_base_vllm_temp0_8:model=allenai/olmOCR-7B-0225-preview:temperature=0.8:prompt_template=fine_tune:response_template=json --repeats 5 --parallel 50
# stop_server

# Feel free to enable if you want, but qwen2 raw is pretty low scoring
# qwen2_vl_7b using sglang server
# start_server sglang "Qwen/Qwen2-VL-7B-Instruct" --chat-template qwen2-vl --mem-fraction-static 0.7
# python -m olmocr.bench.convert server:name=qwen2_vl_7b:model=Qwen/Qwen2-VL-7B-Instruct:temperature=0.1:prompt_template=full:response_template=plain --repeats 5 --parallel 50
# stop_server

# qwen2.5 works best with vllm for now, in a fresh environment
create_conda_env "vllm" "3.11"
source activate vllm
pip install -e .[bench]
pip install --upgrade vllm==0.8.3


start_server vllm "Qwen/Qwen2.5-VL-7B-Instruct" --max-model-len 8192
python -m olmocr.bench.convert --dir "$BENCH_DIR" server:name=qwen25vl_prompt7:model=Qwen/Qwen2.5-VL-7B-Instruct:temperature=0.1:prompt_template=basic:response_template=plain --parallel 50
stop_server

start_server vllm "reducto/RolmOCR" --max-model-len 8192
python -m olmocr.bench.convert --dir "$BENCH_DIR" rolmocr --parallel 50
stop_server

# TODO: Fix this, I was not able to get it to all install successfully
# Create and activate mineru environment
# create_conda_env "mineru" "3.11"
# source activate mineru

# Install magic-pdf and run benchmarks
# echo "Installing magic-pdf and running mineru benchmarks..."
# pip install -U "magic-pdf[full]==1.2.2" --extra-index-url https://wheels.myhloli.com
# python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# pip install huggingface_hub Pillow paddleocr ultralytics doclayout-yolo pycocotools 
# wget https://github.com/opendatalab/MinerU/raw/master/scripts/download_models_hf.py -O download_models_hf.py
# python download_models_hf.py
# python -m olmocr.bench.convert mineru

# Final cleanup
if [ -n "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
    stop_server
fi

echo "All benchmarks completed successfully."