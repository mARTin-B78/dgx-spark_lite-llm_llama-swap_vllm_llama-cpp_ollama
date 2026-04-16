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
RESULTS_DIR="$(dirname "$(readlink -f "$0")")/test-results/benchmarks"
TIMEOUT=1800

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
FILTERS=()

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
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --runs=*)
            RUNS="${1#*=}"
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

Other options:
  --runs N     Override number of runs (default: 3)
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

    local response
    response=$(curl -s --max-time "$TIMEOUT" \
        -X POST "$LLAMA_SWAP_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n --arg model "$model" '{
            model: $model,
            messages: [{role: "user", content: "Say OK"}],
            max_tokens: 5
        }')" 2>&1) || true

    end=$(date +%s.%N)
    elapsed=$(echo "scale=1; $end - $start" | bc)

    local err
    err=$(echo "$response" | jq -r '.error.message // empty' 2>/dev/null)
    if [[ -n "$err" ]]; then
        log "  ${RED}FAILED to load: $err${NC}"
        return 1
    fi

    log "  Model ready (loaded in ${elapsed}s)"
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
    esac
    log "  Running llama-benchy (pp=$PP  tg=$TG  depth=$DEPTH  runs=$RUNS)..."
    log ""

    # --- Run 1: Save JSON for data parsing ---
    local cmd_json="uvx llama-benchy"
    cmd_json+=" --base-url $LLAMA_SWAP_URL/v1"
    cmd_json+=" --model $model"
    cmd_json+=" --pp $PP"
    cmd_json+=" --tg $TG"
    cmd_json+=" --depth $DEPTH"
    cmd_json+=" --runs $RUNS"
    cmd_json+=" --latency-mode generation"
    cmd_json+=" --no-warmup"
    cmd_json+=" --skip-coherence"
    cmd_json+=" --save-result ${json_file}"
    cmd_json+=" --format json"

    local output exit_code=0
    output=$(eval "$cmd_json" 2>&1) || exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        log "  ${RED}llama-benchy failed:${NC}"
        echo "$output" | tail -15 | tee -a "$REPORT_FILE"
        return 1
    fi

    # --- Run 2: Get the markdown table (for sharing on forums) ---
    local cmd_md="uvx llama-benchy"
    cmd_md+=" --base-url $LLAMA_SWAP_URL/v1"
    cmd_md+=" --model $model"
    cmd_md+=" --pp $PP"
    cmd_md+=" --tg $TG"
    cmd_md+=" --depth $DEPTH"
    cmd_md+=" --runs $RUNS"
    cmd_md+=" --latency-mode generation"
    cmd_md+=" --no-warmup"
    cmd_md+=" --skip-coherence"
    cmd_md+=" --save-result ${md_file}"
    cmd_md+=" --format md"

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
esac
log ""
log "  ${DIM}pp = prompt processing (how fast the model reads your input)${NC}"
log "  ${DIM}tg = token generation  (how fast the model writes its reply)${NC}"
log "  ${DIM}depth = pre-filled context tokens (simulates document size)${NC}"
log ""

# Check llama-benchy is available
if ! uvx llama-benchy --help > /dev/null 2>&1; then
    log "${RED}Error: llama-benchy not available via uvx.${NC}"
    log "Install with: pip install llama-benchy  OR  uv pip install llama-benchy"
    exit 1
fi

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
    log "${BOLD}============================================================${NC}"
    log "${BOLD}  [$IDX/$MODEL_COUNT] $MODEL${NC}"
    log "${BOLD}============================================================${NC}"
    log ""

    # Unload previous model to get a clean measurement
    unload_all

    # Load this model
    if ! warmup_model "$MODEL"; then
        FAIL=$((FAIL + 1))
        SUMMARY_STATUS[$MODEL]="FAIL"
        log ""
        continue
    fi

    # Benchmark it
    if run_benchy "$MODEL"; then
        PASS=$((PASS + 1))
        SUMMARY_STATUS[$MODEL]="OK"

        # Extract numbers for final summary from JSON
        local_json="$RESULTS_DIR/${MODEL//\//_}_${TIMESTAMP}.json"
        if [[ -f "$local_json" ]]; then
            eval "$(python3 <<PYEOF
import json
with open('$local_json') as f:
    data = json.load(f)

# Collect baseline (depth=0) and deepest results
baseline_tg = None
deepest_tg = None
max_depth = 0

for b in data.get('benchmarks', []):
    depth = b.get('context_size', 0)
    pp  = b.get('pp_throughput') or {}
    tg  = b.get('tg_throughput') or {}
    pk  = b.get('peak_throughput') or {}
    e2e = b.get('e2e_ttft') or {}
    pp_mean  = pp.get('mean', 0) or 0
    pp_std   = pp.get('std', 0) or 0
    tg_mean  = tg.get('mean', 0) or 0
    tg_std   = tg.get('std', 0) or 0
    pk_mean  = pk.get('mean', 0) or 0
    e2e_mean = e2e.get('mean', 0) or 0

    if pp_mean == 0 and tg_mean == 0:
        continue

    if depth == 0:
        baseline_tg = tg_mean
        # Use baseline for summary row
        if pp_std > 0.5:
            print(f"SUMMARY_PP['$MODEL']='{pp_mean:.0f} +/-{pp_std:.0f}'")
        else:
            print(f"SUMMARY_PP['$MODEL']='{pp_mean:.0f}'")
        if tg_std > 0.5:
            print(f"SUMMARY_TG['$MODEL']='{tg_mean:.1f} +/-{tg_std:.1f}'")
        else:
            print(f"SUMMARY_TG['$MODEL']='{tg_mean:.1f}'")
        print(f"SUMMARY_PEAK['$MODEL']='{pk_mean:.0f}'")
        if e2e_mean > 0:
            print(f"SUMMARY_TTFT['$MODEL']='{e2e_mean:.0f}'")

    if depth > max_depth:
        max_depth = depth
        deepest_tg = tg_mean

# Calculate degradation
if baseline_tg and deepest_tg and baseline_tg > 0 and max_depth > 0:
    pct = ((deepest_tg - baseline_tg) / baseline_tg) * 100
    print(f"SUMMARY_DEGRADATION['$MODEL']='{pct:+.1f}% @{max_depth//1024}k'")

PYEOF
            )" 2>/dev/null
        fi
    else
        FAIL=$((FAIL + 1))
        SUMMARY_STATUS[$MODEL]="FAIL"
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
log "${BOLD}============================================================${NC}"
