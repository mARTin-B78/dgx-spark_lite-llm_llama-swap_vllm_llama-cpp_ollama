#!/bin/bash
# ram_watcher.sh - Emergency memory monitor

echo "Starting RAM watcher. Trigger threshold: 100MB Available"

while true; do
    # Get available memory in KB
    AVAILABLE_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    
    # 100MB = 102400 KB. Lowered to allow 122B models to use swap during load spikes.
    if [ "$AVAILABLE_KB" -lt 102400 ]; then
        echo "[$(date)] EMERGENCY: RAM dropped to ${AVAILABLE_KB} KB! System crash imminent. Executing emergency wipe." >> /home/sparky/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/overnight_benchmark.log
        
        # Instantly kill any running vllm or llama-cpp docker containers to free memory
        # CRITICAL FIX: exclude llama-swap, llama.cpp, and ollama base containers
        docker ps --format '{{.Names}}' | grep -iE 'vllm|llama|nemotron' | grep -vEx 'llama-swap|llama\.cpp|ollama' | xargs -r docker rm -f
        
        echo "[$(date)] Emergency memory purge completed. DGX Spark saved from hard crash." >> /home/sparky/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/overnight_benchmark.log
        
        # Sleep a bit to let the benchmark script catch the error and recover, then resume watching
        sleep 60
    fi
    
    # Check every 3 seconds
    sleep 3
done
