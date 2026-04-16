#!/bin/bash
# =============================================================================
# test-all-models.sh — Smoke-test every llama-swap model
#
# For each model: unload -> send prompt -> measure load time, generation time,
# tokens/sec. Results are printed live and saved to a CSV + summary.
# =============================================================================

set -euo pipefail

# --------------- CONFIG ---------------
LLAMA_SWAP_URL="${LLAMA_SWAP_URL:-http://localhost:28080}"
PROMPT="Explain what a neural network is in exactly three sentences."
MAX_TOKENS=150
TIMEOUT=1800          # 30 min max per model (large models need time to load)
UNLOAD_WAIT=5         # seconds to wait after unloading
RESULTS_DIR="$(dirname "$0")/test-results"
# --------------------------------------

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$RESULTS_DIR/results_${TIMESTAMP}.csv"
LOG_FILE="$RESULTS_DIR/results_${TIMESTAMP}.log"

# CSV header
echo "model,status,load_and_generate_s,prompt_tokens,completion_tokens,total_tokens,tokens_per_sec,first_content" > "$CSV_FILE"

# --------------- HELPERS ---------------
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

escape_csv() {
    # Escape double quotes and wrap in quotes for CSV
    local val="$1"
    val="${val//\"/\"\"}"
    echo "\"$val\""
}

unload_all() {
    curl -sf -X POST "$LLAMA_SWAP_URL/unload" > /dev/null 2>&1 || true
    sleep "$UNLOAD_WAIT"
}

test_model() {
    local model="$1"
    local start end elapsed
    local response http_code body
    local prompt_tok comp_tok total_tok tps
    local content status

    # Send chat completion request, capture body + http code
    start=$(date +%s.%N)

    response=$(curl -s --max-time "$TIMEOUT" -w "\n__HTTP_CODE__%{http_code}" \
        -X POST "$LLAMA_SWAP_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(jq -n \
            --arg model "$model" \
            --arg prompt "$PROMPT" \
            --argjson max_tokens "$MAX_TOKENS" \
            '{
                model: $model,
                messages: [{role: "user", content: $prompt}],
                max_tokens: $max_tokens,
                temperature: 0.7
            }'
        )" 2>&1) || true

    end=$(date +%s.%N)
    elapsed=$(echo "$end - $start" | bc)

    # Split body and http code
    http_code=$(echo "$response" | grep "__HTTP_CODE__" | sed 's/.*__HTTP_CODE__//')
    body=$(echo "$response" | grep -v "__HTTP_CODE__")

    # Check for errors
    if [[ -z "$http_code" || "$http_code" == "000" ]]; then
        status="TIMEOUT"
        log "${RED}  TIMEOUT after ${elapsed}s${NC}"
        echo "$model,TIMEOUT,$elapsed,0,0,0,0,\"timeout\"" >> "$CSV_FILE"
        return 1
    fi

    if [[ "$http_code" != "200" ]]; then
        local err_msg
        err_msg=$(echo "$body" | jq -r '.error.message // .error // "unknown error"' 2>/dev/null || echo "HTTP $http_code")
        status="ERROR"
        log "${RED}  ERROR (HTTP $http_code): $err_msg${NC}"
        echo "$model,ERROR_$http_code,$elapsed,0,0,0,0,$(escape_csv "$err_msg")" >> "$CSV_FILE"
        return 1
    fi

    # Parse success response
    prompt_tok=$(echo "$body" | jq -r '.usage.prompt_tokens // 0' 2>/dev/null)
    comp_tok=$(echo "$body" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
    total_tok=$(echo "$body" | jq -r '.usage.total_tokens // 0' 2>/dev/null)
    content=$(echo "$body" | jq -r '.choices[0].message.content // "no content"' 2>/dev/null)

    # Calculate tokens per second (completion tokens / elapsed time)
    if (( $(echo "$elapsed > 0" | bc -l) )) && [[ "$comp_tok" -gt 0 ]]; then
        tps=$(echo "scale=2; $comp_tok / $elapsed" | bc)
    else
        tps="0"
    fi

    # Truncate content for display (first 80 chars)
    local content_preview="${content:0:80}"
    [[ ${#content} -gt 80 ]] && content_preview="${content_preview}..."

    log "${GREEN}  OK${NC} — ${elapsed}s total | ${comp_tok} tokens | ${BOLD}${tps} tok/s${NC}"
    log "  ${CYAN}Preview:${NC} ${content_preview}"

    # CSV row (first 200 chars of content)
    echo "$model,OK,$elapsed,$prompt_tok,$comp_tok,$total_tok,$tps,$(escape_csv "${content:0:200}")" >> "$CSV_FILE"
    return 0
}

# --------------- MAIN ---------------
log ""
log "${BOLD}============================================================${NC}"
log "${BOLD}  llama-swap Model Test Suite${NC}"
log "${BOLD}============================================================${NC}"
log "  Endpoint:   $LLAMA_SWAP_URL"
log "  Prompt:     \"$PROMPT\""
log "  Max tokens: $MAX_TOKENS"
log "  Timeout:    ${TIMEOUT}s per model"
log "  Results:    $CSV_FILE"
log "${BOLD}============================================================${NC}"
log ""

# Fetch model list
MODELS=$(curl -sf "$LLAMA_SWAP_URL/v1/models" | jq -r '.data[].id' | sort)
MODEL_COUNT=$(echo "$MODELS" | wc -l)

log "Found ${BOLD}${MODEL_COUNT}${NC} models to test."
log ""

# Allow filtering via command-line args
if [[ $# -gt 0 ]]; then
    FILTER="$*"
    log "${YELLOW}Filtering models matching: $FILTER${NC}"
    FILTERED=""
    for m in $MODELS; do
        for f in $FILTER; do
            if [[ "$m" == *"$f"* ]]; then
                FILTERED="${FILTERED}${m}\n"
            fi
        done
    done
    MODELS=$(echo -e "$FILTERED" | grep -v '^$' | sort -u)
    MODEL_COUNT=$(echo "$MODELS" | wc -l)
    log "Testing ${BOLD}${MODEL_COUNT}${NC} matching models."
    log ""
fi

# Counters
PASS=0
FAIL=0
TOTAL_START=$(date +%s.%N)
IDX=0

for MODEL in $MODELS; do
    IDX=$((IDX + 1))
    log "${BOLD}[$IDX/$MODEL_COUNT] $MODEL${NC}"

    # Unload previous model
    log "  Unloading previous model..."
    unload_all

    if test_model "$MODEL"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
    log ""
done

# Final unload
unload_all

TOTAL_END=$(date +%s.%N)
TOTAL_ELAPSED=$(echo "$TOTAL_END - $TOTAL_START" | bc)
TOTAL_MIN=$(echo "scale=1; $TOTAL_ELAPSED / 60" | bc)

# --------------- SUMMARY ---------------
log "${BOLD}============================================================${NC}"
log "${BOLD}  SUMMARY${NC}"
log "${BOLD}============================================================${NC}"
log "  ${GREEN}Passed:${NC} $PASS / $MODEL_COUNT"
log "  ${RED}Failed:${NC} $FAIL / $MODEL_COUNT"
log "  Total time: ${TOTAL_ELAPSED}s (${TOTAL_MIN} min)"
log ""

# Print results table
log "${BOLD}  Model                                              Status  Time(s)  Tok/s${NC}"
log "  $(printf '%.0s-' {1..78})"
while IFS=, read -r model status time ptok ctok ttok tps content; do
    [[ "$model" == "model" ]] && continue  # skip header
    printf -v line "  %-50s %-7s %7s  %5s" "${model:0:50}" "$status" "$time" "$tps"
    if [[ "$status" == "OK" ]]; then
        log "${GREEN}${line}${NC}"
    else
        log "${RED}${line}${NC}"
    fi
done < "$CSV_FILE"

log ""
log "  CSV saved to: ${BOLD}$CSV_FILE${NC}"
log "  Log saved to: ${BOLD}$LOG_FILE${NC}"
log "${BOLD}============================================================${NC}"
