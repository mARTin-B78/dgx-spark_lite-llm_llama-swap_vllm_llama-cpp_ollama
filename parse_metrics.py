import os
import glob
import json
import re

models = [
"Qwen3-Coder-Next-FP8-Dynamic",
"Qwen3-Coder-Next-int4-AutoRound",
"Qwen3-Omni-30B-A3B-Instruct",
"Qwen3-VL-30B-A3B-Instruct-FP8",
"Qwen3.5-4B-Q4_K_M",
"Qwen3.5-27B-Uncensored-DFlash-NVFP4",
"Qwen3.5-35B-A3B-FP8",
"Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M",
"Qwen3.5-122B-A10B-int4-AutoRound",
"Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP",
"Qwen3.6-35B-A3B-PrismaQuant-4.75bit",
"Qwen3.6-27B-PrismaSCOUT-NVFP4",
"Qwen3.6-35B-A3B-FP8"
]

bench_dir = "/home/sparky/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/test-results/benchmarks"
quality_dir = "/home/sparky/Docker/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama/test-results/quality"

print("| Model | Quality Score | Tok/s | Context | Load Time |")
print("|---|---|---|---|---|")

for model in models:
    # Find latest benchmark JSON
    json_files = glob.glob(f"{bench_dir}/{model}_*.json")
    json_files.sort(reverse=True)
    
    tok_s = "N/A"
    context = "N/A"
    
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            benchmarks = data.get("benchmarks", [])
            for b in benchmarks:
                if b.get("context_size", 0) == 0:
                    tok_s = f"{b['tg_throughput']['mean']:.1f}"
                if b.get("context_size", 0) > 0:
                    context = f"{b['context_size']}"
                    
    load_time = "N/A"
    txt_files = glob.glob(f"{bench_dir}/*.txt")
    txt_files.sort(reverse=True)
    for txt in txt_files:
        with open(txt, 'r') as f:
            content = f.read()
            if model in content:
                match = re.search(r"loaded in (\d+\.\d+)s", content)
                if match:
                    load_time = f"{float(match.group(1)):.1f}s"
                    break
                    
    quality = "N/A"
    md_files = glob.glob(f"{quality_dir}/**/*.md", recursive=True)
    md_files.sort(reverse=True)
    for md in md_files:
        with open(md, 'r') as f:
            content = f.read()
            if model in content:
                match = re.search(r"Overall Score:\s*(\d+)/100", content, re.IGNORECASE)
                if match:
                    quality = f"{match.group(1)}/100"
                    break
                    
    print(f"| {model} | {quality} | {tok_s} | {context} | {load_time} |")
