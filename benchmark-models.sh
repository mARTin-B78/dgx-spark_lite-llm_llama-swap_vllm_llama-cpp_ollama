#!/bin/bash
# =============================================================================
# benchmark-models.sh — Benchmark all llama-swap models using llama-benchy
#
# Uses llama-benchy (https://github.com/eugr/llama-benchy) for standardized
# LLM performance measurement. Results are comparable with other DGX Spark
# users who use the same tool.
#
# Benchmark profiles simulate real-world document analysis workloads:
#
#   "Medium Log"  (default)  — 50-page document, pp2048 + tg128 @ depth 16384
#   "Massive Log" (--stress) — 100+ page log,    pp2048 + tg128 @ depth 32768
#   "Extreme"     (--extreme)— 200+ page corpus,  pp2048 + tg128 @ depth 65535
#
# Each profile includes a depth=0 baseline so you can see the performance
# delta as unified memory pressure increases.
#
# Usage:
#   ./benchmark-models.sh                        # Medium Log (default)
#   ./benchmark-models.sh --stress               # Medium + Massive Log
#   ./benchmark-models.sh --extreme              # All three depth levels
#   ./benchmark-models.sh --quick Nemotron       # Fast smoke test
#   ./benchmark-models.sh --runs 5 Qwen3.5-35B   # Custom run count
#   ./benchmark-models.sh Qwen3.5 Nemotron       # Only matching models
# =============================================================================

set -euo pipefail

LLAMA_SWAP_URL="${LLAMA_SWAP_URL:-http://localhost:28080}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
RESULTS_DIR="$SCRIPT_DIR/test-results/benchmarks"
ARENA_BEST_FILE="$SCRIPT_DIR/test-results/arena-best-results.json"
TIMEOUT=1800

# spark-arena-cli: installed to ~/.local/bin on first --arena run
SPARK_CLI="${HOME}/.local/bin/spark-arena-cli"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Defaults: "Medium Log" baseline
PP="2048"
TG="128"
DEPTH="0 16384"
RUNS=3
MODE="medium-log"
CONCURRENCY=""   # space-separated list; empty = not passed to llama-benchy
ARENA_DIR=""     # set by --arena mode
FILTERS=()

# Crash-resume tracking
CHECKPOINT_DIR="$SCRIPT_DIR/test-results/checkpoints"
LAST_SESSION_FILE="$SCRIPT_DIR/test-results/.last-session"
CHECKPOINT_FILE=""
RESUME=false
SKIP_MODELS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            PP="512"
            TG="128"
            DEPTH="0"
            RUNS=1
            MODE="quick"
            shift
            ;;
        --stress)
            # Medium Log + Massive Log
            PP="2048"
            TG="128"
            DEPTH="0 16384 32768"
            RUNS=3
            MODE="stress"
            shift
            ;;
        --extreme)
            # All three: Medium + Massive + Extreme limit
            PP="2048"
            TG="128"
            DEPTH="0 16384 32768 65535"
            RUNS=3
            MODE="extreme"
            shift
            ;;
        --full)
            # Comprehensive sweep (original broad test)
            PP="512 2048"
            TG="128 256 512"
            DEPTH="0 16384 32768"
            RUNS=3
            MODE="full"
            shift
            ;;
        --arena)
            # Spark-Arena leaderboard submission profile
            # https://spark-arena.com/admin
            PP="2048"
            TG="128"
            DEPTH="0 4096 8192 16384 32768 65535 100000"
            CONCURRENCY="1 2 5 10"
            RUNS=3
            MODE="arena"
            shift
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --runs=*)
            RUNS="${1#*=}"
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --help|-h)
            cat <<'HELPEOF'
Usage: benchmark-models.sh [OPTIONS] [FILTER...]

Benchmark your llama-swap models using llama-benchy.
Results use the standard llama-benchy format for comparison
with other DGX Spark users on the NVIDIA forums.

Profiles (simulating real-world document workloads):

  (default)    "Medium Log" — pp2048, tg128 @ depth 0 + 16384
               Simulates a ~50-page document. The depth=0 baseline shows
               raw speed; depth=16384 shows the cost of a full KV cache.

  --stress     "Massive Log" — adds depth 32768
               Doubles the context to simulate a massive error log.
               Watch for tg tok/s drop vs. the medium baseline — that's
               where shared memory bandwidth starts to bottleneck.

  --extreme    "Extreme Limit" — adds depth 65535
               Pushes to ~100 pages of text. Strictly to see if the
               system can process it without crashing or heavy swap.

  --quick      Smoke test — pp512, tg128, depth 0, 1 run
  --full       Broad sweep — pp512+2048, tg128+256+512, depths 0-32k
  --arena      Spark-Arena leaderboard profile — exact spec from spark-arena.com/admin
               Saves results.csv + recipe.yaml per model to test-results/arena-submission/
               Depths: 0 4096 8192 16384 32768 65535 100000 | Concurrency: 1 2 5 10

Other options:
  --runs N     Override number of runs (default: 3)
  --resume     Resume from the last interrupted session (skip already-completed models)
  --help       Show this help

Filters:
  Add model name fragments to only test matching models.
  Example: ./benchmark-models.sh --stress Qwen3.5 Nemotron

Environment:
  LLAMA_SWAP_URL  llama-swap endpoint (default: http://localhost:28080)

What the numbers mean:
  pp tok/s  = Prompt Processing speed. How fast the model reads your input.
              Higher is better. Typically 500-5000+ tok/s on DGX Spark.
  tg tok/s  = Token Generation speed. How fast the model writes its reply.
              Higher is better. This is the number you "feel" when chatting.
              Typically 15-50+ tok/s on DGX Spark depending on model size.
  TTFT      = Time To First Token. The delay before the model starts replying.
              Lower is better. Measured in milliseconds.

Key insight:
  Compare tg tok/s across depths. A big drop from depth=16384 to depth=32768
  means you've found the unified memory bandwidth bottleneck between the ARM
  CPU and the Blackwell GPU on DGX Spark.
HELPEOF
            exit 0
            ;;
        *)
            FILTERS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="$RESULTS_DIR/report_${TIMESTAMP}.txt"

if [[ "$MODE" == "arena" ]]; then
    ARENA_DIR="$SCRIPT_DIR/test-results/arena-submission/${TIMESTAMP}"
    mkdir -p "$ARENA_DIR"
fi

log() { echo -e "$1" | tee -a "$REPORT_FILE"; }

unload_all() {
    curl -sf -X POST "$LLAMA_SWAP_URL/unload" > /dev/null 2>&1 || true
    sleep 5
}

# Warm up: send a tiny request to make llama-swap load the model
warmup_model() {
    local model="$1"
    log "  Loading model via llama-swap..."
    local start end elapsed
    start=$(date +%s.%N)

    local response_file="/tmp/response_${TIMESTAMP}.json"
    local http_code
    http_code=$(curl -s -w "%{http_code}" -o "$response_file" --max-time "$TIMEOUT" \
        -X POST "$LLAMA_SWAP_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "$model" '{
            model: $model,
            messages: [{role: "user", content: "Write a short python hello world script."}],
            max_tokens: 50
        }')" 2>/dev/null) || http_code=0

    end=$(date +%s.%N)
    elapsed=$(echo "scale=1; $end - $start" | bc)

    if [[ "$http_code" -ne 200 ]]; then
        local err_msg
        err_msg=$(jq -r '.error.message // empty' "$response_file" 2>/dev/null)
        if [[ -n "$err_msg" ]]; then
            log "  ${RED}FAILED to load: $err_msg (HTTP $http_code)${NC}"
        else
            log "  ${RED}FAILED to load (HTTP $http_code)${NC}"
        fi
        rm -f "$response_file"
        return 1
    fi

    local content
    content=$(jq -r '(.choices[0].message.reasoning_content // "") + (.choices[0].message.reasoning // "") + (.choices[0].message.content // "")' "$response_file" 2>/dev/null | tr -d '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    rm -f "$response_file"

    if [[ -z "$content" ]]; then
        log "  ${CYAN}Coherence check:${NC} ${RED}FAILED — empty response${NC}"
        WARMUP_FAIL_REASON="EMPTY"
        return 1
    fi

    # Detect repetition loop: split into non-empty words, check if any single word
    # makes up >60% of total words (e.g. "n8n n8n n8n..." or "the the the...")
    # grep -v '^$' filters empty tokens from multi-space code in reasoning_content.
    local word_count most_freq_count most_freq_word
    word_count=$(echo "$content" | tr ' ' '\n' | grep -v '^$' | wc -l)
    if [[ "$word_count" -ge 5 ]]; then
        most_freq_word=$(echo "$content" | tr ' ' '\n' | grep -v '^$' | sort | uniq -c | sort -rn | awk 'NR==1{print $2}')
        most_freq_count=$(echo "$content" | tr ' ' '\n' | grep -v '^$' | grep -cFx "$most_freq_word" 2>/dev/null || echo 0)
        local pct=$(( most_freq_count * 100 / word_count ))
        if [[ "$pct" -ge 60 ]]; then
            log "  ${CYAN}Coherence check:${NC} ${RED}FAILED — repetition loop (\"${most_freq_word}\" = ${pct}% of output)${NC}"
            log "  ${RED}Skipping benchmark — model output is incoherent.${NC}"
            WARMUP_FAIL_REASON="INCOHERENT"
            return 1
        fi
    fi

    log "  ${CYAN}Coherence check:${NC} \"${content:0:150}\""
    log "  Model ready (loaded in ${elapsed}s)"
    WARMUP_FAIL_REASON=""
    return 0
}

# Run llama-benchy and capture results
run_benchy() {
    local model="$1"
    local safe_name="${model//\//_}"
    local json_file="$RESULTS_DIR/${safe_name}_${TIMESTAMP}.json"
    local md_file="$RESULTS_DIR/${safe_name}_${TIMESTAMP}.md"

    # Show profile description
    case "$MODE" in
        medium-log) log "  Profile: Medium Log (50-page document baseline)" ;;
        stress)     log "  Profile: Massive Log (stress test — watch for bandwidth bottleneck)" ;;
        extreme)    log "  Profile: Extreme Limit (push to ~100 pages, crash/swap detection)" ;;
        quick)      log "  Profile: Quick smoke test" ;;
        full)       log "  Profile: Full comprehensive sweep" ;;
        arena)      log "  Profile: Spark-Arena leaderboard (7 depths × 4 concurrency levels)" ;;
    esac
    local concurrency_display=""
    [[ -n "$CONCURRENCY" ]] && concurrency_display="  concurrency=$CONCURRENCY"
    log "  Running llama-benchy (pp=$PP  tg=$TG  depth=$DEPTH  runs=$RUNS${concurrency_display})..."
    log ""

    # Build shared base flags (used by all runs)
    local base_flags=""
    base_flags+=" --base-url $LLAMA_SWAP_URL/v1"
    base_flags+=" --model $model"
    base_flags+=" --pp $PP"
    base_flags+=" --tg $TG"
    base_flags+=" --depth $DEPTH"
    base_flags+=" --runs $RUNS"
    base_flags+=" --latency-mode generation"
    base_flags+=" --no-warmup"
    base_flags+=" --skip-coherence"
    [[ -n "$CONCURRENCY" ]] && base_flags+=" --concurrency $CONCURRENCY"
    [[ "$MODE" == "arena" ]] && base_flags+=" --enable-prefix-caching"

    # --- Run 1: Save JSON for data parsing ---
    local cmd_json="${BENCHY_CMD}${base_flags} --save-result ${json_file} --format json"

    local output exit_code=0
    output=$(eval "$cmd_json" 2>&1) || exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        log "  ${RED}llama-benchy failed:${NC}"
        echo "$output" | tail -15 | tee -a "$REPORT_FILE"
        return 1
    fi

    # --- Arena mode: save submission CSV (separate run, same params) ---
    if [[ "$MODE" == "arena" && -n "$ARENA_DIR" ]]; then
        local arena_model_dir="$ARENA_DIR/${safe_name}"
        mkdir -p "$arena_model_dir"
        local csv_file="$arena_model_dir/results.csv"
        local cmd_csv="${BENCHY_CMD}${base_flags} --save-result ${csv_file} --format csv"
        log "  ${CYAN}Saving arena submission CSV...${NC}"
        local csv_output csv_exit=0
        csv_output=$(eval "$cmd_csv" 2>&1) || csv_exit=$?
        if [[ $csv_exit -ne 0 ]]; then
            log "  ${YELLOW}Warning: CSV run failed — JSON data still saved${NC}"
        else
            log "  ${DIM}Arena CSV  : $csv_file${NC}"
        fi
    fi

    # --- Run 2: Get the markdown table (for sharing on forums) ---
    local cmd_md="${BENCHY_CMD}${base_flags} --save-result ${md_file} --format md"

    local md_output
    md_output=$(eval "$cmd_md" 2>&1) || true

    # Show the full saved markdown file (the format people share on forums)
    if [[ -f "$md_file" ]]; then
        log "  ${CYAN}llama-benchy results (copy this to share on forums):${NC}"
        log ""
        cat "$md_file" | tee -a "$REPORT_FILE"
        log ""
    else
        # Fallback: show table lines from stdout if file wasn't created
        local table_lines
        table_lines=$(echo "$md_output" | grep -E '^\|')
        if [[ -n "$table_lines" ]]; then
            log "  ${CYAN}llama-benchy results:${NC}"
            log ""
            echo "$table_lines" | tee -a "$REPORT_FILE"
            log ""
        fi
    fi

    # --- Parse JSON and show friendly explanation ---
    if [[ -f "$json_file" ]]; then
        python3 <<PYEOF | tee -a "$REPORT_FILE"
import json

with open('$json_file') as f:
    data = json.load(f)

benchmarks = data.get('benchmarks', [])
if not benchmarks:
    print("  (no benchmark data found)")
    exit(0)

# Check if all benchmarks have null results (model failed silently)
all_null = all(
    b.get('pp_throughput') is None and b.get('tg_throughput') is None
    for b in benchmarks
)
if all_null:
    print("  (model returned no usable results — it may not support this benchmark)")
    exit(0)

# Group benchmarks by depth for comparison
depth_results = {}
for b in benchmarks:
    depth = b.get('context_size', 0)
    pp_obj  = b.get('pp_throughput') or {}
    tg_obj  = b.get('tg_throughput') or {}
    pk_obj  = b.get('peak_throughput') or {}
    e2e_obj = b.get('e2e_ttft') or {}
    pp_mean  = pp_obj.get('mean', 0) or 0
    tg_mean  = tg_obj.get('mean', 0) or 0
    if pp_mean > 0 or tg_mean > 0:
        depth_results[depth] = {
            'pp_mean': pp_mean, 'pp_std': (pp_obj.get('std', 0) or 0),
            'tg_mean': tg_mean, 'tg_std': (tg_obj.get('std', 0) or 0),
            'pk_mean': (pk_obj.get('mean', 0) or 0),
            'e2e_mean': (e2e_obj.get('mean', 0) or 0),
        }

# Depth label mapping
depth_labels = {
    0: "Baseline (no context)",
    16384: "Medium Log (~50 pages)",
    32768: "Massive Log (~100 pages)",
    65535: "Extreme Limit (~200 pages)",
}

print("  Context Depth Analysis:")
print("")

baseline_tg = None
for depth in sorted(depth_results.keys()):
    r = depth_results[depth]
    label = depth_labels.get(depth, f"depth {depth}")
    pp_mean, pp_std = r['pp_mean'], r['pp_std']
    tg_mean, tg_std = r['tg_mean'], r['tg_std']
    pk_mean = r['pk_mean']
    e2e_mean = r['e2e_mean']

    print(f"  --- depth={depth:,} — {label} ---")

    # Reading speed
    if pp_mean > 0:
        if pp_std > 0.5:
            print(f"    Reading speed:  {pp_mean:,.0f} +/- {pp_std:,.0f} tok/s")
        else:
            print(f"    Reading speed:  {pp_mean:,.0f} tok/s")

    # Writing speed
    if tg_mean > 0:
        if tg_std > 0.5:
            print(f"    Writing speed:  {tg_mean:.1f} +/- {tg_std:.1f} tok/s  (peak: {pk_mean:.0f})")
        else:
            print(f"    Writing speed:  {tg_mean:.1f} tok/s  (peak: {pk_mean:.0f})")

        # Show degradation from baseline
        if baseline_tg is None:
            baseline_tg = tg_mean
        elif baseline_tg > 0:
            pct = ((tg_mean - baseline_tg) / baseline_tg) * 100
            if pct < -15:
                print(f"    {chr(9888)}  {pct:+.1f}% vs baseline — BANDWIDTH BOTTLENECK DETECTED")
            elif pct < -5:
                print(f"      -> {pct:+.1f}% vs baseline (moderate slowdown)")
            else:
                print(f"      -> {pct:+.1f}% vs baseline (minimal impact)")

        # What it feels like
        if tg_mean >= 40:
            feel = "Very fast — feels instant, smooth streaming"
        elif tg_mean >= 25:
            feel = "Fast — comfortable for interactive chat"
        elif tg_mean >= 15:
            feel = "Good — readable streaming with slight pauses"
        elif tg_mean >= 8:
            feel = "Moderate — noticeable wait, but usable"
        else:
            feel = "Slow — may feel sluggish for chat"
        print(f"      -> {feel}")

    # TTFT
    if e2e_mean > 0:
        print(f"    Time to first token:  {e2e_mean:.0f}ms")
        if e2e_mean < 200:
            print(f"      -> Feels instant")
        elif e2e_mean < 500:
            print(f"      -> Barely noticeable delay")
        elif e2e_mean < 2000:
            print(f"      -> Short pause before response starts")
        else:
            print(f"      -> Noticeable wait ({e2e_mean/1000:.1f}s)")
    print("")

# Summary comparison if multiple depths
if len(depth_results) > 1 and baseline_tg and baseline_tg > 0:
    max_depth = max(depth_results.keys())
    deepest_tg = depth_results[max_depth]['tg_mean']
    total_pct = ((deepest_tg - baseline_tg) / baseline_tg) * 100
    print(f"  Overall impact: depth 0 -> {max_depth:,} = {total_pct:+.1f}% generation speed")
    if total_pct < -20:
        print(f"  Conclusion: Significant unified memory bandwidth bottleneck at depth {max_depth:,}")
    elif total_pct < -10:
        print(f"  Conclusion: Moderate bandwidth pressure — usable but noticeably slower")
    else:
        print(f"  Conclusion: Model handles deep context well on this hardware")
    print("")

PYEOF
    fi

    log "  ${DIM}JSON data : $json_file${NC}"
    log "  ${DIM}Forum table: $md_file${NC}"
    return 0
}

# ---------------------------------------------------------------------------
# spark-arena-cli helpers
# ---------------------------------------------------------------------------

# Download spark-arena-cli binary if it isn't already on PATH / ~/.local/bin
install_spark_arena_cli() {
    if command -v spark-arena-cli &>/dev/null; then
        SPARK_CLI="$(command -v spark-arena-cli)"
        return 0
    fi
    if [[ -x "$SPARK_CLI" ]]; then
        return 0
    fi
    local arch
    arch=$(uname -m)
    local bin_name
    [[ "$arch" == "aarch64" || "$arch" == "arm64" ]] \
        && bin_name="spark-arena-cli-0.1.0-linux-arm64" \
        || bin_name="spark-arena-cli-0.1.0-linux-amd64"
    local url="https://github.com/spark-arena/spark-arena-cli/releases/download/v0.1.0/${bin_name}"
    log "  ${DIM}Downloading spark-arena-cli from GitHub releases...${NC}"
    mkdir -p "$(dirname "$SPARK_CLI")"
    if curl -fsSL "$url" -o "$SPARK_CLI" 2>/dev/null; then
        chmod +x "$SPARK_CLI"
        log "  ${DIM}Installed to $SPARK_CLI${NC}"
        # Add to PATH for this session
        export PATH="$(dirname "$SPARK_CLI"):$PATH"
    else
        log "  ${YELLOW}Warning: could not download spark-arena-cli — manual install needed${NC}"
        SPARK_CLI=""
    fi
}

# Read our personal best tg tok/s (depth=0, concurrency=1) for a model from the history file
get_personal_best_tg() {
    local model="$1"
    if [[ ! -f "$ARENA_BEST_FILE" ]]; then echo "0"; return; fi
    python3 -c "
import json, sys
try:
    d = json.load(open('$ARENA_BEST_FILE'))
    entry = d.get('$model', {})
    print(entry.get('tg_mean', 0))
except:
    print(0)
" 2>/dev/null || echo "0"
}

# Save current result as personal best for a model
save_personal_best() {
    local model="$1"
    local json_file="$2"
    python3 - "$model" "$json_file" "$ARENA_BEST_FILE" <<'PYEOF'
import json, sys, os
model, bench_json, best_file = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    data = json.load(open(bench_json))
except Exception as e:
    sys.exit(0)

# Find depth=0, concurrency=1 entry
baseline = None
for b in data.get("benchmarks", []):
    if b.get("context_size") == 0 and b.get("concurrency") == 1:
        baseline = b
        break
if not baseline:
    # Fallback: first entry with context_size=0
    for b in data.get("benchmarks", []):
        if b.get("context_size") == 0:
            baseline = b
            break
if not baseline:
    sys.exit(0)

tg  = (baseline.get("tg_throughput")  or {}).get("mean", 0) or 0
pp  = (baseline.get("pp_throughput")  or {}).get("mean", 0) or 0
e2e = (baseline.get("e2e_ttft")       or {}).get("mean", 0) or 0

try:
    best = json.load(open(best_file)) if os.path.exists(best_file) else {}
except:
    best = {}

best[model] = {
    "tg_mean": round(tg, 2),
    "pp_mean": round(pp, 1),
    "ttft_ms": round(e2e, 1),
    "timestamp": data.get("timestamp", ""),
    "depth": 0,
    "concurrency": 1,
}
with open(best_file, "w") as f:
    json.dump(best, f, indent=2)
PYEOF
}

# Compare current result vs personal best; if better, print submission info
check_and_suggest_submit() {
    local model="$1"
    local json_file="$2"
    local recipe_file="$3"

    local prev_best
    prev_best=$(get_personal_best_tg "$model")

    local current_tg
    current_tg=$(python3 -c "
import json
data = json.load(open('$json_file'))
for b in data.get('benchmarks', []):
    if b.get('context_size') == 0 and b.get('concurrency') == 1:
        tg = (b.get('tg_throughput') or {}).get('mean', 0) or 0
        print(round(tg, 2))
        exit()
for b in data.get('benchmarks', []):
    if b.get('context_size') == 0:
        tg = (b.get('tg_throughput') or {}).get('mean', 0) or 0
        print(round(tg, 2))
        exit()
print(0)
" 2>/dev/null || echo "0")

    local is_better=0
    python3 -c "exit(0 if float('$current_tg') > float('$prev_best') else 1)" 2>/dev/null && is_better=1

    if [[ "$is_better" -eq 1 ]]; then
        if python3 -c "exit(0 if float('$prev_best') > 0 else 1)" 2>/dev/null; then
            log "  ${GREEN}★ NEW PERSONAL BEST${NC} — ${current_tg} tg tok/s (was ${prev_best})"
        else
            log "  ${GREEN}★ FIRST ARENA RESULT${NC} — ${current_tg} tg tok/s @ depth=0 concurrency=1"
        fi
        save_personal_best "$model" "$json_file"

        # Try to get leaderboard context (best-effort scrape — may be empty)
        log "  ${CYAN}Submission candidate${NC} — consider submitting to spark-arena.com/leaderboard"
        log ""

        install_spark_arena_cli

        if [[ -n "$SPARK_CLI" && -x "$SPARK_CLI" ]]; then
            local is_logged_in=0
            "$SPARK_CLI" benchmark --help &>/dev/null && {
                # Probe login state: spark-arena-cli prints a warning if not configured
                local probe
                probe=$(echo "exit" | timeout 3 "$SPARK_CLI" 2>&1 || true)
                echo "$probe" | grep -q "Warning: Configuration not found" || is_logged_in=1
            }

            if [[ "$is_logged_in" -eq 1 ]]; then
                log "  ${GREEN}spark-arena-cli is logged in.${NC} Run this to submit officially:"
                log "  ${BOLD}  $SPARK_CLI benchmark $recipe_file${NC}"
                log "  ${DIM}  (This re-runs the benchmark via sparkrun and auto-uploads results)${NC}"
            else
                log "  ${YELLOW}spark-arena-cli installed but not logged in.${NC} To submit:"
                log "  ${DIM}  1. $SPARK_CLI login         # authenticate via Google/GitHub${NC}"
                log "  ${DIM}  2. $SPARK_CLI setup         # configure sparkrun + llama-benchy${NC}"
                log "  ${DIM}  3. $SPARK_CLI benchmark $recipe_file${NC}"
            fi
        else
            log "  ${DIM}To submit to spark-arena, install spark-arena-cli:${NC}"
            log "  ${DIM}  curl -fsSL https://github.com/spark-arena/spark-arena-cli/releases/download/v0.1.0/spark-arena-cli-0.1.0-linux-amd64 -o ~/.local/bin/spark-arena-cli && chmod +x ~/.local/bin/spark-arena-cli${NC}"
            log "  ${DIM}  spark-arena-cli login${NC}"
            log "  ${DIM}  spark-arena-cli benchmark $recipe_file${NC}"
        fi
        log ""
    else
        log "  ${DIM}tg ${current_tg} tok/s — personal best is ${prev_best} tok/s (no improvement, skipping submission)${NC}"
    fi
}

# Generate a spark-arena recipe.yaml for a model
generate_recipe_yaml() {
    local model="$1"
    local safe_name="${model//\//_}"
    local out_dir="$ARENA_DIR/${safe_name}"
    mkdir -p "$out_dir"

    python3 - "$model" "$out_dir/recipe.yaml" <<'PYEOF'
import sys, textwrap

model_name = sys.argv[1]
out_path   = sys.argv[2]

# Per-model recipe metadata.
# container: the local Docker image tag we actually use.
# hf_model: canonical HuggingFace model ID for the submission.
RECIPES = {
    "Qwen3.5-35B-A3B-FP8": {
        "hf_model":    "Qwen/Qwen3.5-35B-A3B-Instruct",
        "description": "Qwen3.5 35B MoE FP8-dynamic — reasoning + tool use with MTP-2 speculation",
        "container":   "vllm-node:Version_1",
        "tp": 1, "gpu_mem": 0.7, "max_len": 131072,
        "env": {
            "VLLM_MARLIN_USE_ATOMIC_ADD": "1",
            "VLLM_ENABLE_CUDAGRAPH_GC": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "1",
        },
        "extras": [
            "--kv-cache-dtype fp8",
            "--load-format fastsafetensors",
            "--attention-backend FLASHINFER",
            "--enable-prefix-caching",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens 4096",
            '--speculative-config \'{"method":"mtp","num_speculative_tokens":2}\'',
            "--enable-auto-tool-choice",
            "--tool-call-parser qwen3_xml",
            "--reasoning-parser qwen3",
        ],
    },
    "Qwen3.5-122B-A10B-int4-AutoRound": {
        "hf_model":    "Qwen/Qwen3.5-122B-A10B-Instruct",
        "description": "Qwen3.5 122B MoE INT4 AutoRound — large hybrid reasoning model",
        "container":   "vllm-node-tf5:latest",
        "tp": 1, "gpu_mem": 0.75, "max_len": 40960,
        "env": {"VLLM_MARLIN_USE_ATOMIC_ADD": "1"},
        "extras": [
            "--trust-remote-code",
            "--enforce-eager",
            "--kv-cache-dtype fp8",
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser qwen3_xml",
            "--reasoning-parser qwen3",
        ],
    },
    "Qwen3-VL-30B-A3B-Instruct-FP8": {
        "hf_model":    "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "description": "Qwen3-VL 30B MoE FP8 — vision-language model",
        "container":   "spark-vllm:Version_1",
        "tp": 1, "gpu_mem": 0.60, "max_len": 32768,
        "env": {},
        "extras": [
            "--trust-remote-code",
            "--kv-cache-dtype fp8",
            "--load-format fastsafetensors",
            "--enable-prefix-caching",
            "--limit-mm-per-prompt '{\"image\": 2}'",
            "--enable-auto-tool-choice",
            "--tool-call-parser qwen3_coder",
        ],
    },
    "Qwen3-Omni-30B-A3B-Instruct": {
        "hf_model":    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "description": "Qwen3-Omni 30B MoE — audio + vision + text multimodal",
        "container":   "vllm-node:Version_1",
        "tp": 1, "gpu_mem": 0.75, "max_len": 32768,
        "env": {},
        "extras": [
            "--trust-remote-code",
            "--load-format fastsafetensors",
            "--enable-prefix-caching",
            "--limit-mm-per-prompt '{\"image\": 2, \"audio\": 2}'",
            "--enable-auto-tool-choice",
            "--tool-call-parser qwen3_coder",
        ],
    },
    "Qwen3-Coder-Next-FP8-Dynamic": {
        "hf_model":    "Qwen/Qwen3-Coder-480B-A22B-FP8-Dynamic",
        "description": "Qwen3-Coder-Next 480B MoE FP8-Dynamic — coding specialist",
        "container":   "vllm-node-tf5:latest",
        "tp": 1, "gpu_mem": 0.75, "max_len": 32768,
        "env": {},
        "extras": [
            "--kv-cache-dtype fp8",
            "--load-format fastsafetensors",
            "--attention-backend flashinfer",
            "--enable-auto-tool-choice",
            "--tool-call-parser qwen3_coder",
        ],
    },
    "Qwen3-Coder-Next-int4-AutoRound": {
        "hf_model":    "Qwen/Qwen3-Coder-480B-A22B-Instruct",
        "description": "Qwen3-Coder-Next 480B MoE INT4 AutoRound — coding + throughput optimized",
        "container":   "vllm-node-tf5:latest",
        "tp": 1, "gpu_mem": 0.6, "max_len": 32768,
        "env": {
            "VLLM_MARLIN_USE_ATOMIC_ADD": "1",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_USE_FLASHINFER_MOE_FP8": "1",
        },
        "extras": [
            "--language-model-only",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens 49152",
            "--max-num-seqs 384",
            "--kv-cache-dtype fp8",
            "--load-format fastsafetensors",
            "--optimization-level 3",
            "--performance-mode throughput",
            "--enable-auto-tool-choice",
            "--tool-call-parser qwen3_coder",
        ],
    },
    "Nemotron-3-Nano-4B-FP8": {
        "hf_model":    "nvidia/Nemotron-3-Nano-4B-Instruct",
        "description": "Nemotron-3-Nano 4B FP8 — ultra-fast orchestrator / routing model",
        "container":   "spark-vllm:Version_1",
        "tp": 1, "gpu_mem": 0.5, "max_len": 8192,
        "env": {},
        "extras": [
            "--kv-cache-dtype fp8",
            "--enforce-eager",
            "--trust-remote-code",
            "--load-format fastsafetensors",
            "--enable-prefix-caching",
            "--tool-call-parser qwen3_coder",
            "--reasoning-parser nemotron_v3",
            "--enable-auto-tool-choice",
        ],
    },
    "Nemotron-3-Nano-30B-A3B-NVFP4": {
        "hf_model":    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        "description": "Nemotron-3-Nano 30B NVFP4 — Blackwell-native MoE with nano_v3 reasoning",
        "container":   "vllm-node:Version_1",
        "tp": 1, "gpu_mem": 0.65, "max_len": 131072,
        "env": {
            "VLLM_USE_FLASHINFER_MOE_FP4": "1",
            "VLLM_FLASHINFER_MOE_BACKEND": "throughput",
        },
        "extras": [
            "--kv-cache-dtype fp8",
            "--enforce-eager",
            "--trust-remote-code",
            "--quantization modelopt_fp4",
            "--enable-auto-tool-choice",
            "--tool-call-parser qwen3_coder",
            "--reasoning-parser nano_v3",
        ],
    },
    "Nemotron-3-Super-120B-A12B-NVFP4": {
        "hf_model":    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "description": "Nemotron-3-Super 120B NVFP4 — large reasoning model with CUTLASS MoE",
        "container":   "spark-vllm:Version_1",
        "tp": 1, "gpu_mem": 0.7, "max_len": 65536,
        "env": {
            "VLLM_FLASHINFER_ALLREDUCE_BACKEND": "trtllm",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        },
        "extras": [
            "--kv-cache-dtype fp8",
            "--moe-backend cutlass",
            "--trust-remote-code",
            "--enable-prefix-caching",
            "--load-format fastsafetensors",
            "--tool-call-parser qwen3_coder",
            "--enable-auto-tool-choice",
            "--reasoning-parser nemotron_v3",
        ],
    },
    "GPT-OSS-120B": {
        "hf_model":    "openai/gpt-oss-120b",
        "description": "OpenAI GPT-OSS 120B MXFP4 — open-weights GPT model",
        "container":   "vllm-node-mxfp4",
        "tp": 1, "gpu_mem": 0.7, "max_len": 65536,
        "env": {"VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8": "1"},
        "extras": [
            "--quantization mxfp4",
            "--kv-cache-dtype fp8",
            "--max-num-batched-tokens 8192",
            "--enable-prefix-caching",
            "--load-format fastsafetensors",
            "--tool-call-parser openai",
            "--reasoning-parser openai_gptoss",
            "--enable-auto-tool-choice",
        ],
    },
    "Mistral-Small-24B-Instruct-2501": {
        "hf_model":    "mistralai/Mistral-Small-24B-Instruct-2501",
        "description": "Mistral Small 24B — fast roleplay and instruction following",
        "container":   "vllm-node:Version_1",
        "tp": 1, "gpu_mem": 0.7, "max_len": 32768,
        "env": {},
        "extras": [
            "--trust-remote-code",
            "--enforce-eager",
            "--enable-auto-tool-choice",
            "--tool-call-parser mistral",
        ],
    },
    "Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M-GGUF": {
        "hf_model":    "HauhauCS/Qwen3.5-35B-A3B-Uncensored-Aggressive-GGUF",
        "description": "Qwen3.5 35B MoE Q4_K_M GGUF — uncensored, llama.cpp serving",
        "container":   "ghcr.io/martin-b78/llama-cpp-spark:latest",
        "tp": 1, "gpu_mem": 0.0, "max_len": 16384,
        "env": {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"},
        "extras": [
            "--ctx-size 16384",
            "--n-gpu-layers 99",
            "--parallel 4",
            "--no-mmap",
        ],
        "runtime": "llama-cpp",
    },
}

r = RECIPES.get(model_name)
if not r:
    # Unknown model — generate a minimal placeholder recipe
    r = {
        "hf_model":    f"<TODO: HuggingFace model ID for {model_name}>",
        "description": model_name,
        "container":   "<TODO: container image>",
        "tp": 1, "gpu_mem": 0.7, "max_len": 32768,
        "env": {}, "extras": [],
    }

runtime  = r.get("runtime", "vllm")
hf_model = r["hf_model"]
extras   = "\n".join(f"    {e} \\" for e in r["extras"])
env_block = ""
if r["env"]:
    env_lines = "\n".join(f"  {k}: '{v}'" for k, v in r["env"].items())
    env_block = f"env:\n{env_lines}\n"

if runtime == "llama-cpp":
    cmd = (
        f"llama-server \\\n"
        f"    -hf {hf_model} \\\n"
        f"    --host {{host}} --port {{port}} \\\n"
        + "\n".join(f"    {e} \\" for e in r["extras"])
        + "\n"
    )
else:
    cmd = (
        f"vllm serve {hf_model} \\\n"
        f"    --served-model-name {model_name} \\\n"
        f"    --host {{host}} --port {{port}} \\\n"
        f"    --tensor-parallel-size {{tensor_parallel}} \\\n"
        f"    --gpu-memory-utilization {{gpu_memory_utilization}} \\\n"
        f"    --max-model-len {{max_model_len}} \\\n"
        + "\n".join(f"    {e} \\" for e in r["extras"])
        + "\n"
    )

yaml = f"""recipe_version: '1'
name: {model_name}
description: {r['description']}
model: {hf_model}
cluster_only: false
container: {r['container']}
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: {r['tp']}
  gpu_memory_utilization: {r['gpu_mem']}
  max_model_len: {r['max_len']}
{env_block}command: |
  {cmd.rstrip()}
solo_only: false
"""

with open(out_path, "w") as f:
    f.write(yaml)

print(f"  Recipe YAML: {out_path}")
PYEOF

    if [[ $? -ne 0 ]]; then
        log "  ${YELLOW}Warning: recipe.yaml generation failed for $model${NC}"
    fi
}

# ---------------------------------------------------------------------------
# Crash-resume checkpoint functions
# ---------------------------------------------------------------------------

# Create a new checkpoint file for this session and register it as last-known
init_checkpoint() {
    mkdir -p "$CHECKPOINT_DIR"
    CHECKPOINT_FILE="$CHECKPOINT_DIR/session_${TIMESTAMP}.json"
    python3 - "$CHECKPOINT_FILE" "$MODE" "$PP" "$TG" "$DEPTH" "$RUNS" <<'PYEOF'
import json, sys, datetime
f, mode, pp, tg, depth, runs = sys.argv[1:]
data = {
    "session_id": f.split("_")[-1].replace(".json", ""),
    "started_at": datetime.datetime.now().isoformat(),
    "mode": mode,
    "settings": {"pp": pp, "tg": tg, "depth": depth, "runs": int(runs)},
    "models": [],
}
with open(f, "w") as fh:
    json.dump(data, fh, indent=2)
PYEOF
    echo "$CHECKPOINT_FILE" > "$LAST_SESSION_FILE"
}

# Mark a model as started (written BEFORE warmup so a crash is detectable)
checkpoint_model_start() {
    local model="$1"
    [[ -z "$CHECKPOINT_FILE" ]] && return
    python3 - "$CHECKPOINT_FILE" "$model" <<'PYEOF'
import json, sys, datetime
f, model = sys.argv[1:]
with open(f) as fh:
    data = json.load(fh)
data["models"].append({
    "model": model,
    "status": "started",
    "started_at": datetime.datetime.now().isoformat(),
})
with open(f, "w") as fh:
    json.dump(data, fh, indent=2)
PYEOF
}

# Mark a model as completed with its result (OK / FAIL / INCOHERENT / EMPTY)
checkpoint_model_done() {
    local model="$1"
    local result="$2"
    [[ -z "$CHECKPOINT_FILE" ]] && return
    python3 - "$CHECKPOINT_FILE" "$model" "$result" <<'PYEOF'
import json, sys, datetime
f, model, result = sys.argv[1:]
with open(f) as fh:
    data = json.load(fh)
for entry in reversed(data["models"]):
    if entry["model"] == model and entry.get("status") == "started":
        entry["status"] = "completed"
        entry["result"] = result
        entry["completed_at"] = datetime.datetime.now().isoformat()
        break
with open(f, "w") as fh:
    json.dump(data, fh, indent=2)
PYEOF
}

# Load a previous checkpoint and populate SKIP_MODELS (models already done)
load_resume_checkpoint() {
    if [[ ! -f "$LAST_SESSION_FILE" ]]; then
        echo -e "${RED}Error: no previous session found. Run without --resume first.${NC}" >&2
        exit 1
    fi
    local prev_cp
    prev_cp=$(cat "$LAST_SESSION_FILE")
    if [[ ! -f "$prev_cp" ]]; then
        echo -e "${RED}Error: checkpoint file not found: $prev_cp${NC}" >&2
        exit 1
    fi

    log "  ${CYAN}Resuming from checkpoint:${NC} $prev_cp"

    local info
    info=$(python3 - "$prev_cp" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
completed = [e["model"] for e in data.get("models", []) if e.get("status") == "completed"]
last_started = next(
    (e["model"] for e in reversed(data.get("models", [])) if e.get("status") == "started"),
    "",
)
s = data.get("settings", {})
print("MODE:" + data.get("mode", ""))
print("SETTINGS:pp=" + str(s.get("pp","")) + "  tg=" + str(s.get("tg","")) +
      "  depth=" + str(s.get("depth","")) + "  runs=" + str(s.get("runs","")))
print("LAST_STARTED:" + last_started)
for m in completed:
    print("DONE:" + m)
PYEOF
)
    while IFS= read -r line; do
        case "$line" in
            DONE:*)    SKIP_MODELS+=("${line#DONE:}") ;;
            MODE:*)    log "  Previous mode     : ${line#MODE:}" ;;
            SETTINGS:*) log "  Previous settings : ${line#SETTINGS:}" ;;
            LAST_STARTED:*)
                local last="${line#LAST_STARTED:}"
                if [[ -n "$last" ]]; then
                    log "  ${YELLOW}Crashed while running: $last${NC} — will re-run it."
                fi
                ;;
        esac
    done <<< "$info"

    log "  Skipping ${#SKIP_MODELS[@]} already-completed model(s)."
    log ""
}

# --------------- MAIN ---------------
log ""
log "${BOLD}============================================================${NC}"
log "${BOLD}  DGX Spark Model Benchmark${NC}"
log "${BOLD}  powered by llama-benchy (github.com/eugr/llama-benchy)${NC}"
log "${BOLD}============================================================${NC}"
log ""
log "  Endpoint : $LLAMA_SWAP_URL"
log "  Mode     : $MODE"
log "  Settings : pp=$PP  tg=$TG  depth=$DEPTH  runs=$RUNS"
log "  Date     : $(date '+%Y-%m-%d %H:%M')"
log "  Results  : $RESULTS_DIR"
log ""

# Mode descriptions
case "$MODE" in
    medium-log)
        log "  ${CYAN}Profile: Medium Log Baseline${NC}"
        log "  ${DIM}Simulates a ~50-page document (depth=16384). Establishes your${NC}"
        log "  ${DIM}baseline generation speed with a moderately full KV cache.${NC}"
        ;;
    stress)
        log "  ${CYAN}Profile: Massive Log Stress Test${NC}"
        log "  ${DIM}Doubles context to simulate a massive error log (depth=32768).${NC}"
        log "  ${DIM}Watch for tg tok/s drop — that's the unified memory bottleneck.${NC}"
        ;;
    extreme)
        log "  ${CYAN}Profile: Extreme Limit Test${NC}"
        log "  ${DIM}Pushes to ~200 pages (depth=65535). Tests if the system can${NC}"
        log "  ${DIM}process it without crashing or heavy swap paging.${NC}"
        ;;
    quick)
        log "  ${DIM}Quick smoke test — just checking if models respond.${NC}"
        ;;
    full)
        log "  ${DIM}Full comprehensive sweep — broad pp/tg/depth combinations.${NC}"
        ;;
    arena)
        log "  ${CYAN}Profile: Spark-Arena Leaderboard Submission${NC}"
        log "  ${DIM}Official spark-arena.com benchmark spec: 7 depth levels × 4 concurrency${NC}"
        log "  ${DIM}levels × 3 runs = 84 data points per model.${NC}"
        log "  ${DIM}Generates results.csv + recipe.yaml per model in:${NC}"
        log "  ${DIM}  $ARENA_DIR${NC}"
        ;;
esac
log ""
log "  ${DIM}pp = prompt processing (how fast the model reads your input)${NC}"
log "  ${DIM}tg = token generation  (how fast the model writes its reply)${NC}"
log "  ${DIM}depth = pre-filled context tokens (simulates document size)${NC}"
log ""

# Initialize crash-resume checkpoint and, if --resume, load previous state
init_checkpoint
[[ "$RESUME" == true ]] && load_resume_checkpoint

# Detect how llama-benchy is available: uvx (preferred), direct, or missing
if uvx llama-benchy --help > /dev/null 2>&1; then
    BENCHY_CMD="uvx llama-benchy"
elif command -v llama-benchy > /dev/null 2>&1; then
    BENCHY_CMD="llama-benchy"
else
    log "${RED}Error: llama-benchy not found.${NC}"
    log ""
    log "Install one of these ways:"
    log "  ${BOLD}uvx${NC} (no install needed):  works if 'uv' is installed"
    log "  ${BOLD}pip install llama-benchy${NC}  (install into current venv/system)"
    log "  ${BOLD}uv pip install llama-benchy${NC}"
    exit 1
fi
log "  ${DIM}Using: $BENCHY_CMD${NC}"

# Fetch model list
MODELS=$(curl -sf "$LLAMA_SWAP_URL/v1/models" | jq -r '.data[].id' | sort)
MODEL_COUNT=$(echo "$MODELS" | wc -l)

# Apply filters
if [[ ${#FILTERS[@]} -gt 0 ]]; then
    log "Filtering models matching: ${FILTERS[*]}"
    FILTERED=""
    for m in $MODELS; do
        for f in "${FILTERS[@]}"; do
            if [[ "$m" == *"$f"* ]]; then
                FILTERED="${FILTERED}${m}\n"
            fi
        done
    done
    MODELS=$(echo -e "$FILTERED" | grep -v '^$' | sort -u)
    MODEL_COUNT=$(echo "$MODELS" | wc -l)
fi

log "Found ${BOLD}${MODEL_COUNT}${NC} model(s) to benchmark."
log ""

PASS=0
FAIL=0
IDX=0
TOTAL_START=$(date +%s.%N)

# Collect results for final summary
declare -A SUMMARY_PP SUMMARY_TG SUMMARY_PEAK SUMMARY_TTFT SUMMARY_STATUS SUMMARY_DEGRADATION

for MODEL in $MODELS; do
    IDX=$((IDX + 1))

    # Skip models already completed in a --resume session
    if [[ "$RESUME" == true ]]; then
        _skip=false
        for _done in "${SKIP_MODELS[@]:-}"; do
            [[ "$MODEL" == "$_done" ]] && { _skip=true; break; }
        done
        if [[ "$_skip" == true ]]; then
            log "  ${DIM}⏭  [$IDX/$MODEL_COUNT] $MODEL — skipped (completed in previous session)${NC}"
            continue
        fi
    fi

    log "${BOLD}============================================================${NC}"
    log "${BOLD}  [$IDX/$MODEL_COUNT] $MODEL${NC}"
    log "${BOLD}============================================================${NC}"
    log ""

    # Write checkpoint BEFORE starting so a crash is detectable
    checkpoint_model_start "$MODEL"

    # Unload previous model to get a clean measurement
    unload_all

    # Load this model
    WARMUP_FAIL_REASON=""
    if ! warmup_model "$MODEL"; then
        FAIL=$((FAIL + 1))
        SUMMARY_STATUS[$MODEL]="${WARMUP_FAIL_REASON:-FAIL}"
        checkpoint_model_done "$MODEL" "${WARMUP_FAIL_REASON:-FAIL}"
        log ""
        continue
    fi

    # Benchmark it
    if run_benchy "$MODEL"; then
        PASS=$((PASS + 1))
        SUMMARY_STATUS[$MODEL]="OK"
        checkpoint_model_done "$MODEL" "OK"
        # In arena mode: generate recipe YAML and check vs personal best
        if [[ "$MODE" == "arena" ]]; then
            generate_recipe_yaml "$MODEL" 2>&1 | tee -a "$REPORT_FILE"
            local arena_json="$RESULTS_DIR/${MODEL//\//_}_${TIMESTAMP}.json"
            local arena_recipe="$ARENA_DIR/${MODEL//\//_}/recipe.yaml"
            check_and_suggest_submit "$MODEL" "$arena_json" "$arena_recipe" 2>&1 | tee -a "$REPORT_FILE"
        fi

        # Extract numbers for final summary from JSON
        local_json="$RESULTS_DIR/${MODEL//\//_}_${TIMESTAMP}.json"
        if [[ -f "$local_json" ]]; then
            # Extract metrics using jq for the summary table
            # We look for depth 0 (baseline) and the deepest result
            baseline_tg=$(jq -r '.benchmarks[] | select(.context_size == 0) | .tg_throughput.mean // empty' "$local_json" | head -n1)
            max_depth=$(jq -r '.benchmarks[].context_size' "$local_json" | sort -rn | head -n1)
            deepest_tg=$(jq -r ".benchmarks[] | select(.context_size == $max_depth) | .tg_throughput.mean // empty" "$local_json" | head -n1)
            
            # Baseline metrics for the summary table
            SUMMARY_PP[$MODEL]=$(jq -r '.benchmarks[] | select(.context_size == 0) | if .pp_throughput.std > 0.5 then "\(.pp_throughput.mean + 0.5 | floor) +/-\(.pp_throughput.std + 0.5 | floor)" else "\(.pp_throughput.mean + 0.5 | floor)" end' "$local_json" | head -n1)
            SUMMARY_TG[$MODEL]=$(jq -r '.benchmarks[] | select(.context_size == 0) | if .tg_throughput.std > 0.5 then "\((.tg_throughput.mean * 10 + 0.5 | floor) / 10) +/-\((.tg_throughput.std * 10 + 0.5 | floor) / 10)" else "\((.tg_throughput.mean * 10 + 0.5 | floor) / 10)" end' "$local_json" | head -n1)
            SUMMARY_PEAK[$MODEL]=$(jq -r '.benchmarks[] | select(.context_size == 0) | .peak_throughput.mean + 0.5 | floor' "$local_json" | head -n1)
            SUMMARY_TTFT[$MODEL]=$(jq -r '.benchmarks[] | select(.context_size == 0) | .e2e_ttft.mean + 0.5 | floor' "$local_json" | head -n1)
            
            if [[ -n "$baseline_tg" && -n "$deepest_tg" && "$max_depth" -gt 0 ]]; then
                pct=$(echo "scale=1; (($deepest_tg - $baseline_tg) / $baseline_tg) * 100" | bc)
                SUMMARY_DEGRADATION[$MODEL]="${pct}% @$((max_depth/1024))k"
            fi
        fi
    else
        FAIL=$((FAIL + 1))
        SUMMARY_STATUS[$MODEL]="FAIL"
        checkpoint_model_done "$MODEL" "FAIL"
    fi
    log ""
done

# Final cleanup
unload_all

TOTAL_END=$(date +%s.%N)
TOTAL_ELAPSED=$(echo "$TOTAL_END - $TOTAL_START" | bc)
TOTAL_MIN=$(echo "scale=1; $TOTAL_ELAPSED / 60" | bc)

# =============================================
# FINAL REPORT
# =============================================
log ""
log "${BOLD}============================================================${NC}"
log "${BOLD}  BENCHMARK REPORT — DGX Spark${NC}"
log "${BOLD}============================================================${NC}"
log ""
log "  Date: $(date '+%Y-%m-%d %H:%M')  |  Mode: $MODE  |  Runs: $RUNS"
log "  Depths tested: $DEPTH"
log "  Models tested: $MODEL_COUNT  |  Passed: ${GREEN}$PASS${NC}  Failed: ${RED}$FAIL${NC}"
log "  Total benchmark time: ${TOTAL_MIN} min"
log "  Checkpoint : $CHECKPOINT_FILE"
[[ $FAIL -gt 0 ]] && log "  ${YELLOW}Tip: if this was interrupted, resume with: ./benchmark-models.sh --resume${NC}"
log ""
log "  ${BOLD}$(printf '%-42s  %14s  %12s  %8s  %8s  %14s' 'Model' 'Read (pp)' 'Write (tg)' 'Peak' 'TTFT' 'Deep ctx')${NC}"
log "  ${DIM}$(printf '%-42s  %14s  %12s  %8s  %8s  %14s' '' 'tok/s' 'tok/s' 'tok/s' 'ms' 'degradation')${NC}"
log "  $(printf '%.0s-' {1..106})"

for MODEL in $MODELS; do
    local_name="$MODEL"
    [[ ${#local_name} -gt 42 ]] && local_name="${local_name:0:39}..."

    status="${SUMMARY_STATUS[$MODEL]:-FAIL}"
    pp="${SUMMARY_PP[$MODEL]:-—}"
    tg="${SUMMARY_TG[$MODEL]:-—}"
    peak="${SUMMARY_PEAK[$MODEL]:-—}"
    ttft="${SUMMARY_TTFT[$MODEL]:-—}"
    degrad="${SUMMARY_DEGRADATION[$MODEL]:-—}"

    if [[ "$status" == "OK" ]]; then
        printf -v line "  %-42s  %14s  %12s  %8s  %8s  %14s" "$local_name" "$pp" "$tg" "$peak" "$ttft" "$degrad"
        log "${GREEN}${line}${NC}"
    elif [[ "$status" == "INCOHERENT" ]]; then
        printf -v line "  %-42s  %14s  %12s  %8s  %8s  %14s" "$local_name" "INCOHERENT" "—" "—" "—" "—"
        log "${YELLOW}${line}${NC}"
    else
        printf -v line "  %-42s  %14s  %12s  %8s  %8s  %14s" "$local_name" "FAIL" "—" "—" "—" "—"
        log "${RED}${line}${NC}"
    fi
done

log ""
log "  ${DIM}---------------------------------------------------------------${NC}"
log "  ${DIM}How to read this table:${NC}"
log "  ${DIM}${NC}"
log "  ${DIM}  Read (pp)    = How fast the model reads your prompt (higher = better).${NC}"
log "  ${DIM}  Write (tg)   = How fast the model types its answer at depth=0 baseline.${NC}"
log "  ${DIM}                  This is the speed you feel when chatting.${NC}"
log "  ${DIM}                  Humans read at ~4 tok/s, so 20+ feels smooth.${NC}"
log "  ${DIM}  Peak         = Fastest burst speed observed in a 1-second window.${NC}"
log "  ${DIM}  TTFT         = Time until the first word appears (lower = better).${NC}"
log "  ${DIM}  Deep ctx     = Speed change at max tested depth vs baseline.${NC}"
log "  ${DIM}                  >-15% = unified memory bandwidth bottleneck.${NC}"
log "  ${DIM}---------------------------------------------------------------${NC}"
log ""
log "  Results saved to:"
log "    Report     : $REPORT_FILE"
log "    JSON data  : $RESULTS_DIR/*_${TIMESTAMP}.json"
log "    Forum tables: $RESULTS_DIR/*_${TIMESTAMP}.md"
log ""
log "  ${DIM}Tip: To share on NVIDIA forums, copy the llama-benchy tables${NC}"
log "  ${DIM}from the .md files — they use the standard format everyone knows.${NC}"

if [[ "$MODE" == "arena" && -n "$ARENA_DIR" ]]; then
    log ""
    log "${BOLD}============================================================${NC}"
    log "${BOLD}  SPARK-ARENA SUBMISSION FILES${NC}"
    log "${BOLD}============================================================${NC}"
    log ""
    log "  Submission directory: ${CYAN}$ARENA_DIR${NC}"
    log ""
    log "  Per-model folders (one submission per model):"
    for MODEL in $MODELS; do
        safe="${MODEL//\//_}"
        model_dir="$ARENA_DIR/$safe"
        if [[ -f "$model_dir/results.csv" && -f "$model_dir/recipe.yaml" ]]; then
            log "    ${GREEN}✓${NC} $MODEL"
            log "        recipe.yaml : $model_dir/recipe.yaml"
            log "        results.csv : $model_dir/results.csv"
        elif [[ "${SUMMARY_STATUS[$MODEL]:-FAIL}" != "OK" ]]; then
            log "    ${RED}✗${NC} $MODEL  (benchmark failed — no submission files)"
        fi
    done
    log ""
    log "  ${BOLD}How to submit to spark-arena.com/admin:${NC}"
    log "  ${DIM}1. Open https://spark-arena.com/admin${NC}"
    log "  ${DIM}2. For each model folder above:${NC}"
    log "  ${DIM}     a. Paste or upload the contents of recipe.yaml${NC}"
    log "  ${DIM}     b. Upload results.csv${NC}"
    log "  ${DIM}3. Submit one entry per model.${NC}"
    log ""
    log "  ${DIM}Note: The recipe.yaml 'model:' field uses the canonical HuggingFace ID.${NC}"
    log "  ${DIM}If your model was downloaded from a different source, update it.${NC}"
fi
log "${BOLD}============================================================${NC}"
