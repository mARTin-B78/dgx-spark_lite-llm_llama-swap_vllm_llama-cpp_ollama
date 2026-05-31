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

# Quality (tool-eval-bench) — opt-in second pass after llama-benchy, same load
QUALITY=false
QUALITY_ONLY=false            # skip llama-benchy, only run tool-eval-bench
QUALITY_MODE="short"          # short | full | hardmode
QUALITY_CATEGORIES=""         # optional letters, e.g. "K A J"
QUALITY_SEED="42"             # default seed
QUALITY_PRESSURE=""           # --context-pressure
QUALITY_PRESSURE_SWEEP=""     # --context-pressure-sweep
QUALITY_CONTEXT_SIZE=""       # --context-size
QUALITY_EXTRA_ARGS=""         # catch-all for tunneled args
QUALITY_DIR="$SCRIPT_DIR/test-results/quality"
TOOLEVAL_CMD=""

# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------
# When the script is invoked with NO arguments and stdin is a TTY, show a
# guided menu instead of dumping a wall of help. The wizard fetches the live
# model list from llama-swap, lets the user pick models + tests, then
# rewrites the positional args so the normal arg-parser below handles the
# rest. Pass --no-wizard or any flag to skip.
# ---------------------------------------------------------------------------

HAS_GUM=false
command -v gum >/dev/null 2>&1 && HAS_GUM=true

_wiz_say() { printf '\033[1m%s\033[0m\n' "$1"; }
_wiz_dim() { printf '\033[2m%s\033[0m\n' "$1"; }
_wiz_warn(){ printf '\033[1;33m%s\033[0m\n' "$1"; }

# Pretty step header — mimics the openclaw / charm `huh` look:
#   ◇ Step title
#     value or sub-content
_wiz_step() {
    local title="$1"
    if $HAS_GUM; then
        # Magenta diamond + bold title
        gum style --foreground 212 "◇ $(gum style --bold "$title")" >&2
    else
        printf '\n\033[1;35m◇\033[0m \033[1m%s\033[0m\n' "$title" >&2
    fi
}

# Boxed sub-section (rounded border, indented one level)
_wiz_box() {
    local content="$1"
    if $HAS_GUM; then
        gum style --border rounded --padding "0 1" --margin "0 0 0 4" \
            --border-foreground 240 "$content" >&2
    else
        printf '    \033[2m┌──────────────────────────────────────────\033[0m\n' >&2
        printf '    \033[2m│\033[0m %s\n' "$content" >&2
        printf '    \033[2m└──────────────────────────────────────────\033[0m\n' >&2
    fi
}

# multi-select prompt: prints chosen indices (one per line) on stdout.
# Args: "<header>" "opt1" "opt2" ...
_wiz_multiselect() {
    local header="$1"; shift
    local -a opts=("$@")
    local i

    if $HAS_GUM; then
        # `gum choose --no-limit` returns the chosen strings. We map back to
        # indices. Pre-select everything by passing --selected with all opts.
        local selected_arg
        selected_arg=$(IFS=,; echo "${opts[*]}")
        local picks
        picks=$(printf '%s\n' "${opts[@]}" | gum choose \
            --no-limit \
            --header "$header" \
            --header.foreground=212 \
            --cursor-prefix "[ ] " \
            --selected-prefix "[x] " \
            --unselected-prefix "[ ] " \
            --selected="$selected_arg" \
            --height 20 \
            </dev/tty) || picks=""
        # Map each chosen string back to its index in opts.
        local pick
        while IFS= read -r pick; do
            [[ -z "$pick" ]] && continue
            for i in "${!opts[@]}"; do
                if [[ "${opts[$i]}" == "$pick" ]]; then
                    echo "$i"
                    break
                fi
            done
        done <<<"$picks"
        return
    fi

    # ----- Fallback: numeric toggle UI -----
    local -a sel=()
    for i in "${!opts[@]}"; do sel[i]=1; done

    while true; do
        printf '\n\033[1m%s\033[0m\n' "$header" >&2
        for i in "${!opts[@]}"; do
            if [[ ${sel[i]:-0} -eq 1 ]]; then
                printf '  \033[32m[%2d] [x]\033[0m %s\n' "$((i+1))" "${opts[i]}" >&2
            else
                printf '  [%2d] [ ] %s\n' "$((i+1))" "${opts[i]}" >&2
            fi
        done
        printf '\n  \033[2mEnter numbers to toggle (e.g. "1 3 5"), "a"=all, "n"=none, ENTER=done\033[0m\n' >&2
        local input
        read -r -p "  > " input </dev/tty || input=""
        [[ -z "$input" ]] && break
        case "$input" in
            a|A|all)  for i in "${!opts[@]}"; do sel[i]=1; done ;;
            n|N|none) for i in "${!opts[@]}"; do sel[i]=0; done ;;
            *)
                local n idx
                for n in $input; do
                    [[ "$n" =~ ^[0-9]+$ ]] || continue
                    idx=$((n-1))
                    [[ $idx -lt 0 || $idx -ge ${#opts[@]} ]] && continue
                    if [[ ${sel[idx]:-0} -eq 1 ]]; then sel[idx]=0; else sel[idx]=1; fi
                done
                ;;
        esac
    done
    for i in "${!opts[@]}"; do
        [[ ${sel[i]:-0} -eq 1 ]] && echo "$i"
    done
}

# single-choice prompt: prints chosen INDEX (0-based) on stdout.
# Args: "<header>" <default-index> "opt1" "opt2" ...
_wiz_singlechoice() {
    local header="$1"; local default_idx="$2"; shift 2
    local -a opts=("$@")
    local i

    if $HAS_GUM; then
        local pick
        pick=$(printf '%s\n' "${opts[@]}" | gum choose \
            --header "$header" \
            --header.foreground=212 \
            --selected="${opts[$default_idx]}" \
            --height 12 \
            </dev/tty) || pick=""
        if [[ -z "$pick" ]]; then
            echo "$default_idx"
            return
        fi
        for i in "${!opts[@]}"; do
            if [[ "${opts[$i]}" == "$pick" ]]; then echo "$i"; return; fi
        done
        echo "$default_idx"
        return
    fi

    # ----- Fallback: numbered prompt -----
    local input
    while true; do
        printf '\n\033[1m%s\033[0m\n' "$header" >&2
        for i in "${!opts[@]}"; do
            if [[ $i -eq $default_idx ]]; then
                printf '  \033[32m[%d]\033[0m %s  \033[2m(default)\033[0m\n' "$((i+1))" "${opts[i]}" >&2
            else
                printf '  [%d] %s\n' "$((i+1))" "${opts[i]}" >&2
            fi
        done
        read -r -p "  > " input </dev/tty || input=""
        if [[ -z "$input" ]]; then
            echo "$default_idx"
            return
        fi
        if [[ "$input" =~ ^[0-9]+$ ]] && [[ $input -ge 1 && $input -le ${#opts[@]} ]]; then
            echo "$((input-1))"
            return
        fi
        printf '  \033[1;33m  Please enter 1-%d.\033[0m\n' "${#opts[@]}" >&2
    done
}

# Free-text input. Args: "<prompt>" "<placeholder>"
_wiz_input() {
    local prompt="$1" placeholder="${2:-}"
    if $HAS_GUM; then
        gum input --prompt "  > " --placeholder "$placeholder" --header "$prompt" </dev/tty
    else
        printf '\n\033[1m%s\033[0m\n' "$prompt" >&2
        [[ -n "$placeholder" ]] && printf '  \033[2m(%s)\033[0m\n' "$placeholder" >&2
        local input
        read -r -p "  > " input </dev/tty || input=""
        echo "$input"
    fi
}

# Yes/No confirm. Args: "<prompt>" [default-yes|default-no]
_wiz_confirm() {
    local prompt="$1" default="${2:-default-yes}"
    if $HAS_GUM; then
        if [[ "$default" == "default-no" ]]; then
            gum confirm --default=false "$prompt" </dev/tty
        else
            gum confirm --default=true "$prompt" </dev/tty
        fi
        return $?
    fi
    local input
    local hint="[Y/n]"
    [[ "$default" == "default-no" ]] && hint="[y/N]"
    printf '\n\033[1m%s\033[0m %s ' "$prompt" "$hint" >&2
    read -r input </dev/tty || input=""
    if [[ -z "$input" ]]; then
        [[ "$default" == "default-no" ]] && return 1 || return 0
    fi
    case "$input" in
        y|Y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

run_wizard() {
    # Output goes to STDERR so the function's stdout stays clean for capturing
    # the resulting argv (we use a global array, but stderr-only chrome means
    # nothing visual leaks if a caller ever did capture stdout).
    if $HAS_GUM; then
        gum style \
            --border double --padding "1 3" --margin "1 0" \
            --border-foreground 212 --foreground 212 --bold \
            "DGX Spark Benchmark — Interactive Setup" \
            "" \
            "powered by llama-benchy     (github.com/eugr/llama-benchy)" \
            "powered by tool-eval-bench  (github.com/SeraphimSerapis/tool-eval-bench)" >&2
    else
        {
            printf '\n\033[1m============================================================\033[0m\n'
            printf   '\033[1m  DGX Spark Benchmark — Interactive Setup\033[0m\n'
            printf   '\033[1m  powered by llama-benchy     (github.com/eugr/llama-benchy)\033[0m\n'
            printf   '\033[1m  powered by tool-eval-bench  (github.com/SeraphimSerapis/tool-eval-bench)\033[0m\n'
            printf   '\033[1m============================================================\033[0m\n'
        } >&2
    fi
    {
        printf '  No flags given — launching the guided setup.\n'
        printf '  \033[2m(Pass any flag, e.g. --quick, or BENCH_NO_WIZARD=1 to skip. --help for CLI.)\033[0m\n'
        if ! $HAS_GUM; then
            printf '  \033[2mTip: install \033[0m\033[1mgum\033[0m\033[2m for an arrow-key TUI:\033[0m\n'
            printf '  \033[2m     https://github.com/charmbracelet/gum#installation\033[0m\n'
        fi
    } >&2

    # Fetch live model list from llama-swap.
    local models_json
    models_json=$(curl -sf --max-time 5 "$LLAMA_SWAP_URL/v1/models" 2>/dev/null) || true
    if [[ -z "$models_json" ]]; then
        printf '\n\033[0;31m  Could not reach llama-swap at %s\033[0m\n' "$LLAMA_SWAP_URL" >&2
        printf '  \033[2mStart the stack first (docker compose up -d) and try again.\033[0m\n' >&2
        return 1
    fi
    local -a all_models
    mapfile -t all_models < <(echo "$models_json" | jq -r '.data[].id' | sort)
    if [[ ${#all_models[@]} -eq 0 ]]; then
        printf '\n\033[0;31m  llama-swap returned no models.\033[0m\n' >&2
        return 1
    fi

    # --- 1. Model selection ----------------------------------------------
    _wiz_step "Step 1/4 — Which models? (${#all_models[@]} available)"
    if $HAS_GUM; then
        _wiz_dim "  Use ↑/↓, SPACE to toggle, ENTER to confirm. Defaults to all selected." >&2
    fi
    local -a chosen_idx
    mapfile -t chosen_idx < <(_wiz_multiselect "Pick models (space to toggle)" "${all_models[@]}")
    if [[ ${#chosen_idx[@]} -eq 0 ]]; then
        printf '\n\033[1;33m  No models selected — aborting.\033[0m\n' >&2
        return 1
    fi
    local -a chosen_models=()
    local idx
    for idx in "${chosen_idx[@]}"; do chosen_models+=("${all_models[$idx]}"); done
    _wiz_box "$(printf '%d models selected\n%s' "${#chosen_models[@]}" "$(printf '  • %s\n' "${chosen_models[@]}")")"

    # --- 2. Which tests --------------------------------------------------
    _wiz_step "Step 2/4 — Which tests?"
    local tests_idx
    tests_idx=$(_wiz_singlechoice "Test mode" 0 \
        "Speed only         — llama-benchy (pp/tg/depth)" \
        "Quality only       — tool-eval-bench (tool-call accuracy)" \
        "Speed AND Quality  — both passes on the same loaded model")
    local want_speed=false want_quality=false
    case "$tests_idx" in
        0) want_speed=true ;;
        1) want_quality=true ;;
        2) want_speed=true; want_quality=true ;;
    esac
    local tests_summary=""
    [[ "$want_speed"   == true ]] && tests_summary+="llama-benchy (speed)  "
    [[ "$want_quality" == true ]] && tests_summary+="tool-eval-bench (quality)"
    _wiz_box "$tests_summary"

    # --- 3a. Speed profile (if speed selected) ---------------------------
    local speed_flag="" speed_label=""
    if [[ "$want_speed" == true ]]; then
        _wiz_step "Step 3/4 — Speed profile (llama-benchy)"
        local speed_idx
        speed_idx=$(_wiz_singlechoice "Profile" 0 \
            "Medium Log    — pp2048 tg128 depth=0,16384  3 runs   (default, ~5 min/model)" \
            "Quick smoke   — pp2048 tg128 depth=0,16384  1 run    (fast sanity check)" \
            "Stress        — adds depth=32768  3 runs              (find memory bottleneck)" \
            "Extreme       — adds depth=65535  3 runs              (~200 page corpus)" \
            "Full sweep    — pp512+2048 tg128+256+512 depths 0-32k (broad)" \
            "Arena         — official spark-arena.com profile      (leaderboard submission)")
        case "$speed_idx" in
            0) speed_flag="";          speed_label="Medium Log (default)" ;;
            1) speed_flag="--quick";   speed_label="Quick smoke" ;;
            2) speed_flag="--stress";  speed_label="Stress (+32k)" ;;
            3) speed_flag="--extreme"; speed_label="Extreme (+65k)" ;;
            4) speed_flag="--full";    speed_label="Full sweep" ;;
            5) speed_flag="--arena";   speed_label="Arena profile" ;;
        esac
        _wiz_box "$speed_label"
    fi

    # --- 3b. Quality mode (if quality selected) --------------------------
    local quality_flags=() quality_label=""
    if [[ "$want_quality" == true ]]; then
        _wiz_step "Step 3/4 — Quality mode (tool-eval-bench)"
        local q_idx
        q_idx=$(_wiz_singlechoice "Quality mode" 0 \
            "Short    — 15 core scenarios   (~2-5 min/model)" \
            "Full     — 69 scenarios        (~15-30 min/model)" \
            "Hardmode — full + 5 adversarial scenarios")
        case "$q_idx" in
            0) quality_flags+=("--quality-mode" "short");    quality_label="short" ;;
            1) quality_flags+=("--quality-mode" "full");     quality_label="full" ;;
            2) quality_flags+=("--quality-mode" "hardmode"); quality_label="hardmode" ;;
        esac
        if [[ "$want_speed" == false ]]; then
            quality_flags+=("--quality-only")
        else
            quality_flags+=("--quality")
        fi

        local cats_input
        cats_input=$(_wiz_input "Optional: restrict to specific categories? (letters A-P, space-separated; ENTER=all)" \
                                "e.g. K A J  —  see github.com/SeraphimSerapis/tool-eval-bench#categories")
        if [[ -n "$cats_input" ]]; then
            quality_flags+=("--quality-categories" "$cats_input")
            quality_label+=" (categories: $cats_input)"
        fi
        _wiz_box "$quality_label"
    fi

    # --- 4. Confirm ------------------------------------------------------
    # Build the equivalent CLI command for display + the argv we'll inject.
    local -a built_argv=()
    [[ -n "$speed_flag" ]] && built_argv+=("$speed_flag")
    if [[ ${#quality_flags[@]} -gt 0 ]]; then
        built_argv+=("${quality_flags[@]}")
    fi
    # Filters: if the user picked ALL models, omit filters; otherwise pass
    # exact model names as filters (the existing parser already handles this).
    if [[ ${#chosen_models[@]} -lt ${#all_models[@]} ]]; then
        built_argv+=("${chosen_models[@]}")
    fi

    _wiz_step "Step 4/4 — Review & confirm"
    {
        # Human-readable summary
        printf '  Models  : %d\n' "${#chosen_models[@]}"
        printf '  Tests   : %s\n' "$tests_summary"
        [[ -n "$speed_label"   ]] && printf '  Speed   : %s\n' "$speed_label"
        [[ -n "$quality_label" ]] && printf '  Quality : %s\n' "$quality_label"
        printf '\n  \033[1mEquivalent CLI:\033[0m\n'
        printf '  \033[36m./benchmark-models.sh'
        local a
        for a in "${built_argv[@]}"; do
            if [[ "$a" == *" "* ]]; then printf ' "%s"' "$a"; else printf ' %s' "$a"; fi
        done
        printf '\033[0m\n\n'
    } >&2

    if ! _wiz_confirm "Run this benchmark now?"; then
        printf '\n  Cancelled.\n' >&2
        return 1
    fi

    # Export to global so caller can use `set --`
    WIZARD_ARGV=("${built_argv[@]}")
    return 0
}

# Trigger: only when invoked with no arguments AND we have a real terminal.
# Setting BENCH_NO_WIZARD=1 disables the wizard entirely.
if [[ $# -eq 0 ]] && [[ -t 0 ]] && [[ -t 1 ]] && [[ "${BENCH_NO_WIZARD:-0}" != "1" ]]; then
    if run_wizard; then
        set -- "${WIZARD_ARGV[@]}"
    else
        exit 0
    fi
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            PP="2048"
            TG="128"
            DEPTH="0 16384"
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
        --quality)
            QUALITY=true
            shift
            ;;
        --quality-mode)
            QUALITY=true
            QUALITY_MODE="$2"
            shift 2
            ;;
        --quality-mode=*)
            QUALITY=true
            QUALITY_MODE="${1#*=}"
            shift
            ;;
        --quality-only)
            QUALITY=true
            QUALITY_ONLY=true
            shift
            ;;
        --quality-categories)
            QUALITY=true
            QUALITY_CATEGORIES="$2"
            shift 2
            ;;
        --quality-categories=*)
            QUALITY=true
            QUALITY_CATEGORIES="${1#*=}"
            shift
            ;;
        --seed)
            QUALITY_SEED="$2"
            shift 2
            ;;
        --context-pressure)
            QUALITY_PRESSURE="$2"
            shift 2
            ;;
        --context-pressure-sweep)
            QUALITY_PRESSURE_SWEEP="$2"
            shift 2
            ;;
        --context-size)
            QUALITY_CONTEXT_SIZE="$2"
            shift 2
            ;;
        --quality-args)
            QUALITY_EXTRA_ARGS="$2"
            shift 2
            ;;
        --help|-h)
            cat <<'HELPEOF'
Usage: benchmark-models.sh [OPTIONS] [FILTER...]

If invoked with no arguments on an interactive terminal, an interactive
wizard guides you through model selection and test choice. To skip the
wizard from a pure-CLI workflow, pass any flag (e.g. --quick) or set
BENCH_NO_WIZARD=1.

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
  --quality    After llama-benchy, also run tool-eval-bench (tool-call quality)
               on the same loaded model. Avoids the cost of a second load.
  --quality-only
               Skip llama-benchy entirely. Load the model, measure load
               time, and run tool-eval-bench.
  --quality-mode MODE
               short    (default) — 15 core scenarios, ~2-5 min/model
               full     — 69 scenarios, ~15-30 min/model
               hardmode — full + 5 adversarial scenarios
  --quality-categories "K A J"
               Run only specific tool-eval-bench category letters (A-P).
               Implies --quality.
  --seed N     Random seed for tool-eval-bench (default: 42)
  --context-pressure R
               Set context pressure (0.0-1.0) for tool-eval-bench
  --context-pressure-sweep START-END
               Run a context pressure sweep (e.g. 0.7-1.0)
  --context-size N
               Explicitly set context window size for tool-eval-bench
  --quality-args "ARGS"
               Tunnel raw arguments directly to tool-eval-bench
               Example: --quality-args "--temperature 0.0 --parallel 1"
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
    # Poll until BOTH (a) llama-swap reports no running upstream AND (b) no
    # docker model-runner container is still present. llama-swap removes the
    # container from its /running list as soon as it sends the stop, but
    # docker takes additional time to actually kill the process and release
    # GPU memory. Without waiting for docker, the next warmup races the
    # shutting-down container and llama-swap returns HTTP 502.
    local waited=0 max_wait=120
    while (( waited < max_wait )); do
        local llamaswap_clear=true docker_clear=true

        # (a) llama-swap /running endpoint — empty result means nothing loaded.
        # We treat HTTP failure (curl exit non-zero) as "still loaded" rather
        # than "clear" so a transient 502 doesn't trick us into proceeding.
        local running rc
        running=$(curl -s --max-time 3 -w '\n%{http_code}' "$LLAMA_SWAP_URL/running" 2>/dev/null) || true
        local code body
        code=$(echo "$running" | tail -n1)
        body=$(echo "$running" | sed '$d')
        if [[ "$code" == "200" ]]; then
            if echo "$body" | jq -e '(.running // .) | length == 0' >/dev/null 2>&1; then
                :
            else
                llamaswap_clear=false
            fi
        else
            llamaswap_clear=false
        fi

        # (b) docker — any container started by llama-swap follows the
        # naming pattern vllm-*, llamacpp-*, sglang-* or ollama-* (see
        # llama-swap/config.yaml `docker run --name` lines).
        if docker ps --format '{{.Names}}' 2>/dev/null \
            | grep -qE '^(vllm-|llamacpp-|sglang-|ollama-)' ; then
            docker_clear=false
        fi

        if $llamaswap_clear && $docker_clear; then
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done
    # Final settle so VRAM the docker daemon just released is visible to the
    # next process (the kernel needs a moment to reflect freed device memory).
    sleep 5
}

# Warm up: send a tiny request to make llama-swap load the model
warmup_model() {
    local model="$1"
    log "  Loading model via llama-swap..."
    local start end elapsed
    start=$(date +%s.%N)

    local response_file="/tmp/response_${TIMESTAMP}.json"
    local http_code attempt=1 max_attempts=2
    while (( attempt <= max_attempts )); do
        http_code=$(curl -s -w "%{http_code}" -o "$response_file" --max-time "$TIMEOUT" \
            -X POST "$LLAMA_SWAP_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$(jq -n --arg model "$model" '{
                model: $model,
                messages: [{role: "user", content: "Write a short python hello world script."}],
                max_tokens: 50
            }')" 2>/dev/null) || http_code=0

        # 200 → done. 502 / 0 (no response, gateway down) → transient, retry once
        # after a longer wait. Any other 4xx / 5xx (e.g. 400 OOM) is not
        # transient — fail fast with the message.
        if [[ "$http_code" == "200" ]]; then
            break
        fi
        if (( attempt < max_attempts )) && { [[ "$http_code" == "502" ]] || [[ "$http_code" == "0" ]]; }; then
            log "  ${YELLOW}HTTP $http_code from llama-swap — likely a still-shutting-down container. Retrying in 30s...${NC}"
            unload_all     # extra teardown wait before the second attempt
            sleep 30
            attempt=$((attempt + 1))
            continue
        fi
        break
    done

    end=$(date +%s.%N)
    elapsed=$(echo "scale=1; $end - $start" | bc)

    if [[ "$http_code" -ne 200 ]]; then
        local err_msg
        err_msg=$(jq -r '.error.message // empty' "$response_file" 2>/dev/null)
        if [[ -z "$err_msg" ]]; then
            # Many 502s come back as plain text, not JSON — surface the first
            # line of the body so the actual cause (e.g. "process exited with
            # code 1") is visible in the log instead of just "HTTP 502".
            err_msg=$(head -c 200 "$response_file" 2>/dev/null | tr '\n' ' ' | head -c 200)
        fi
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
# tool-eval-bench (quality) helpers
# ---------------------------------------------------------------------------

# Run tool-eval-bench against the currently-loaded model. Captures the score
# from the markdown report it writes into $QUALITY_DIR. Returns the score as a
# global (QUALITY_SCORE / QUALITY_RATING / QUALITY_REPORT) for the summary.
QUALITY_SCORE=""
QUALITY_RATING=""
QUALITY_REPORT=""
QUALITY_DEPLOY=""             # Deployability subscore
QUALITY_RESPONSE_MS=""        # median turn latency in seconds
QUALITY_CTXPRESS=""           # context pressure %
QUALITY_TOTAL_POINTS=""       # "12 / 12"
QUALITY_CATS=""               # "Tool Selection 100%, Multi-Step Chains 100%"
# Helper to look up max_len for a model. Mirrors the values in the RECIPES
# dict inside generate_recipe_yaml (kept in sync manually — the previous
# sed-extracts-then-exec approach broke on the first inner '}').
get_model_max_len() {
    local model="$1"
    case "$model" in
        Qwen3.5-35B-A3B-FP8)                                          echo 131072 ;;
        Qwen3.5-122B-A10B-int4-AutoRound)                             echo  40960 ;;
        Qwen3-VL-30B-A3B-Instruct-FP8)                                echo  32768 ;;
        Qwen3-Omni-30B-A3B-Instruct)                                  echo  32768 ;;
        Qwen3-Coder-Next-FP8-Dynamic)                                 echo  32768 ;;
        Qwen3-Coder-Next-int4-AutoRound)                              echo  32768 ;;
        Nemotron-3-Nano-4B-FP8)                                       echo   8192 ;;
        Nemotron-3-Nano-30B-A3B-NVFP4)                                echo 131072 ;;
        Nemotron-3-Super-120B-A12B-NVFP4)                             echo  65536 ;;
        GPT-OSS-120B)                                                 echo  65536 ;;
        Mistral-Small-24B-Instruct-2501)                              echo  32768 ;;
        Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M-GGUF)   echo  16384 ;;
        Qwen3.6-35B-A3B-FP8)                                          echo 262144 ;;
        Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive)               echo  16384 ;;
        Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4)                 echo 131072 ;;
        *)                                                            echo "" ;;
    esac
}

run_quality() {
    local model="$1"
    QUALITY_SCORE=""; QUALITY_RATING=""; QUALITY_REPORT=""
    [[ -z "$TOOLEVAL_CMD" ]] && return 0

    log ""
    log "  ${CYAN}Running tool-eval-bench (quality)...${NC}"

    local flags=""
    flags+=" --base-url $LLAMA_SWAP_URL"
    flags+=" --model $model"
    flags+=" --backend vllm"
    flags+=" --no-warmup"
    flags+=" --no-live"

    # Handle context size: explicit flag > auto-detect from recipes
    local ctx_size="$QUALITY_CONTEXT_SIZE"
    if [[ -z "$ctx_size" ]]; then
        ctx_size=$(get_model_max_len "$model")
        [[ -n "$ctx_size" ]] && log "  ${DIM}Auto-detected context size from recipe: $ctx_size${NC}"
    fi
    [[ -n "$ctx_size" ]] && flags+=" --context-size $ctx_size"

    [[ -n "$QUALITY_SEED" ]] && flags+=" --seed $QUALITY_SEED"
    [[ -n "$QUALITY_PRESSURE" ]] && flags+=" --context-pressure $QUALITY_PRESSURE"
    [[ -n "$QUALITY_PRESSURE_SWEEP" ]] && flags+=" --context-pressure-sweep $QUALITY_PRESSURE_SWEEP"
    [[ -n "$QUALITY_EXTRA_ARGS" ]] && flags+=" $QUALITY_EXTRA_ARGS"
    flags+=" --output-dir $QUALITY_DIR"

    case "$QUALITY_MODE" in
        short)    flags+=" --short" ;;
        full)     ;;  # default 69 scenarios
        hardmode) flags+=" --hardmode" ;;
        *)        log "  ${YELLOW}Unknown --quality-mode '$QUALITY_MODE', defaulting to short${NC}"
                  flags+=" --short" ;;
    esac
    [[ -n "$QUALITY_CATEGORIES" ]] && flags+=" --categories $QUALITY_CATEGORIES"

    # Capture stdout + stderr so we can extract the score line.
    local out_file="/tmp/tooleval_${TIMESTAMP}_$$.log"
    local exit_code=0
    eval "$TOOLEVAL_CMD$flags" >"$out_file" 2>&1 || exit_code=$?

    if [[ $exit_code -ne 0 ]]; then
        log "  ${RED}tool-eval-bench failed (exit $exit_code):${NC}"
        tail -20 "$out_file" | tee -a "$REPORT_FILE"
        rm -f "$out_file"
        return 1
    fi

    # tool-eval-bench writes its report to $QUALITY_DIR/<run_id>/report.md
    # and prints a final summary to stdout. Pull score from stdout first
    # ("Final Score: 73.4 / 100" or "Score: 73 ★★★"), fall back to newest
    # report file in $QUALITY_DIR.
    QUALITY_SCORE=$(grep -oE 'Score:?[[:space:]]*[0-9]+(\.[0-9]+)?' "$out_file" \
        | head -n1 | grep -oE '[0-9]+(\.[0-9]+)?' || true)
    QUALITY_RATING=$(grep -oE '★+[^[:space:]]*[[:space:]]*[A-Za-z]+' "$out_file" \
        | head -n1 || true)

    # Locate the report markdown file. tool-eval-bench writes per-run reports
    # named <ISO-timestamp>_<runid>.md (NOT report.md) two subdirectories deep
    # ($QUALITY_DIR/YYYY/MM/<file>.md), so we have to find by mtime.
    local newest_report
    newest_report=$(find "$QUALITY_DIR" -type f -name '*.md' -printf '%T@ %p\n' 2>/dev/null \
        | sort -rn | head -n1 | awk '{print $2}')
    if [[ -z "$QUALITY_SCORE" && -n "$newest_report" && -f "$newest_report" ]]; then
        QUALITY_REPORT="$newest_report"
        QUALITY_SCORE=$(grep -oE 'Score[^0-9]*[0-9]+(\.[0-9]+)?' "$newest_report" \
            | head -n1 | grep -oE '[0-9]+(\.[0-9]+)?' || true)
        QUALITY_RATING=$(grep -oE '★+[^[:space:]]*[[:space:]]*[A-Za-z]+' "$newest_report" \
            | head -n1 || true)
    elif [[ -n "$newest_report" ]]; then
        QUALITY_REPORT="$newest_report"
    fi

    # Echo the tool-eval-bench summary block (last 25 lines of stdout) to BOTH
    # the terminal and the main report file. The previous `>/dev/null` swallowed
    # the nice summary in the terminal, so users only saw the parsed score line.
    log ""
    tail -25 "$out_file" | tee -a "$REPORT_FILE"
    rm -f "$out_file"

    # Parse extra subscores from report.md (Deployability, Responsiveness,
    # Context Pressure, Total Points, Category Scores) so the final summary
    # can show a tool-eval-bench detail block.
    QUALITY_DEPLOY=""; QUALITY_RESPONSE_MS=""; QUALITY_CTXPRESS=""
    QUALITY_TOTAL_POINTS=""; QUALITY_CATS=""
    if [[ -n "$QUALITY_REPORT" && -f "$QUALITY_REPORT" ]]; then
        QUALITY_DEPLOY=$(grep -m1 -oE '\*\*Deployability\*\*:[[:space:]]*\*\*[0-9]+(\.[0-9]+)?\*\*' "$QUALITY_REPORT" \
            | grep -oE '[0-9]+(\.[0-9]+)?' | tail -n1 || true)
        QUALITY_RESPONSE_MS=$(grep -m1 -oE 'median turn:[[:space:]]*[0-9]+(\.[0-9]+)?s' "$QUALITY_REPORT" \
            | grep -oE '[0-9]+(\.[0-9]+)?' || true)
        QUALITY_CTXPRESS=$(grep -m1 -oE '\*\*Context Pressure\*\*:[[:space:]]*[0-9]+%' "$QUALITY_REPORT" \
            | grep -oE '[0-9]+' || true)
        QUALITY_TOTAL_POINTS=$(grep -m1 -oE '\*\*Total Points\*\*:[[:space:]]*[0-9]+[[:space:]]*/[[:space:]]*[0-9]+' "$QUALITY_REPORT" \
            | sed 's/.*Points\*\*:[[:space:]]*//' || true)
        # Category Scores table: rows look like "| Tool Selection | 6 | 6 | 100% |"
        # Pull category-name + percent and join into one inline string.
        QUALITY_CATS=$(awk '
            /^## Category Scores/ {in_cat=1; next}
            in_cat && /^## / {in_cat=0}
            in_cat && /^\| *[A-Za-z]/ && !/^\| *Category/ && !/^\| *---/ {
                # extract first and last cells
                gsub(/^\| */, ""); gsub(/ *\| *$/, "")
                n = split($0, c, / *\| */)
                if (n >= 4) printf "%s%s %s", (sep?", ":""), c[1], c[n]
                sep=1
            }
        ' "$QUALITY_REPORT")
    fi

    if [[ -n "$QUALITY_SCORE" ]]; then
        log "  ${CYAN}Quality score:${NC} ${BOLD}${QUALITY_SCORE}/100${NC} ${QUALITY_RATING}"
        [[ -n "$QUALITY_TOTAL_POINTS" ]] && log "  ${DIM}Points: $QUALITY_TOTAL_POINTS${NC}"
        [[ -n "$QUALITY_DEPLOY" ]] && log "  ${DIM}Deployability: ${QUALITY_DEPLOY}/100  |  Median turn: ${QUALITY_RESPONSE_MS:-?}s  |  Context pressure: ${QUALITY_CTXPRESS:-?}%${NC}"
        [[ -n "$QUALITY_CATS" ]] && log "  ${DIM}Categories: $QUALITY_CATS${NC}"
        [[ -n "$QUALITY_REPORT" ]] && log "  ${DIM}Quality report: $QUALITY_REPORT${NC}"
    else
        log "  ${YELLOW}Quality score could not be parsed from tool-eval-bench output${NC}"
    fi
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
        "container":   "vllm-node:latest",
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
        "container":   "vllm-node:latest",
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
        "container":   "vllm-node:latest",
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
        "container":   "vllm-node:latest",
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
        "container":   "vllm-node:latest",
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
        "container":   "vllm-node:latest",
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
        "container":   "vllm-node:latest",
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
log "${BOLD}  powered by llama-benchy     (github.com/eugr/llama-benchy)${NC}"
log "${BOLD}  powered by tool-eval-bench  (github.com/SeraphimSerapis/tool-eval-bench)${NC}"
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

# Initialize crash-resume checkpoint and, if --resume, load previous state.
# load_resume_checkpoint must run BEFORE init_checkpoint: init_checkpoint
# overwrites .last-session with the new session path, so reading it afterwards
# returns the new (empty) checkpoint instead of the previous one.
[[ "$RESUME" == true ]] && load_resume_checkpoint
init_checkpoint

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

# Detect tool-eval-bench when --quality is on. Uvx preferred; fall back to PATH.
if [[ "$QUALITY" == true ]]; then
    if uvx tool-eval-bench --help > /dev/null 2>&1; then
        TOOLEVAL_CMD="uvx tool-eval-bench"
    elif command -v tool-eval-bench > /dev/null 2>&1; then
        TOOLEVAL_CMD="tool-eval-bench"
    else
        log "${YELLOW}Warning: --quality requested but tool-eval-bench not found.${NC}"
        log "  Install: ${BOLD}uv tool install git+https://github.com/SeraphimSerapis/tool-eval-bench.git${NC}"
        log "  Quality runs will be skipped."
        QUALITY=false
    fi
    if [[ "$QUALITY" == true ]]; then
        mkdir -p "$QUALITY_DIR"
        log "  ${DIM}Quality: $TOOLEVAL_CMD (mode=$QUALITY_MODE${QUALITY_CATEGORIES:+, categories=$QUALITY_CATEGORIES})${NC}"
    fi
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
declare -A SUMMARY_PP SUMMARY_TG SUMMARY_PEAK SUMMARY_TTFT SUMMARY_STATUS SUMMARY_DEGRADATION SUMMARY_QUALITY SUMMARY_LOAD
declare -A SUMMARY_QUALITY_DEPLOY SUMMARY_QUALITY_RESPONSE SUMMARY_QUALITY_CTXPRESS SUMMARY_QUALITY_POINTS SUMMARY_QUALITY_CATS SUMMARY_QUALITY_REPORT SUMMARY_QUALITY_RATING

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
    load_start=$(date +%s.%N)
    if warmup_model "$MODEL"; then
        warmup_ok=true
    else
        warmup_ok=false
    fi
    load_end=$(date +%s.%N)
    SUMMARY_LOAD[$MODEL]=$(printf "%.1fs" "$(echo "$load_end - $load_start" | bc)")

    if [[ "$warmup_ok" != true ]]; then
        FAIL=$((FAIL + 1))
        SUMMARY_STATUS[$MODEL]="${WARMUP_FAIL_REASON:-FAIL}"
        checkpoint_model_done "$MODEL" "${WARMUP_FAIL_REASON:-FAIL}"
        log ""
        continue
    fi

    # Benchmark it
    benchy_ok=true
    if [[ "$QUALITY_ONLY" == true ]]; then
        log "  ${DIM}Skipping llama-benchy (--quality-only)${NC}"
    else
        if run_benchy "$MODEL"; then
            PASS=$((PASS + 1))
            SUMMARY_STATUS[$MODEL]="OK"
            checkpoint_model_done "$MODEL" "OK"
            # In arena mode: generate recipe YAML and check vs personal best
            if [[ "$MODE" == "arena" ]]; then
                generate_recipe_yaml "$MODEL" 2>&1 | tee -a "$REPORT_FILE"
                arena_json="$RESULTS_DIR/${MODEL//\//_}_${TIMESTAMP}.json"
                arena_recipe="$ARENA_DIR/${MODEL//\//_}/recipe.yaml"
                check_and_suggest_submit "$MODEL" "$arena_json" "$arena_recipe" 2>&1 | tee -a "$REPORT_FILE"
            fi
        else
            benchy_ok=false
        fi
    fi

    if [[ "$benchy_ok" == true ]]; then
        # Extract numbers for final summary from JSON
        local_json="$RESULTS_DIR/${MODEL//\//_}_${TIMESTAMP}.json"
        if [[ -f "$local_json" ]]; then
            # Extract metrics using jq for the summary table
            # We look for depth 0 (baseline) and the deepest result
            baseline_tg=$(jq -r '.benchmarks[] | select(.context_size == 0) | .tg_throughput.mean // empty' "$local_json" | head -n1)
            max_depth=$(jq -r '.benchmarks[].context_size' "$local_json" | sort -rn | head -n1)
            deepest_tg=$(jq -r ".benchmarks[] | select(.context_size == $max_depth) | .tg_throughput.mean // empty" "$local_json" | head -n1)
            
            # Baseline metrics for the summary table.
            # Some servers (vLLM in certain modes) don't return prompt_tokens
            # for the depth=0 baseline, so pp_throughput is null. Fall back to
            # the smallest non-null depth so pp doesn't show as 0.
            SUMMARY_PP[$MODEL]=$(jq -r '
                [.benchmarks[]
                  | select(.pp_throughput != null and .pp_throughput.mean != null)
                  | {ctx: .context_size, pp: .pp_throughput}]
                | sort_by(.ctx)
                | (.[0] // empty)
                | if .pp.std > 0.5
                  then "\(.pp.mean + 0.5 | floor) +/-\(.pp.std + 0.5 | floor)"
                  else "\(.pp.mean + 0.5 | floor)"
                  end' "$local_json" | head -n1)
            [[ -z "${SUMMARY_PP[$MODEL]}" ]] && SUMMARY_PP[$MODEL]="—"

            SUMMARY_TG[$MODEL]=$(jq -r '.benchmarks[] | select(.context_size == 0) | select(.tg_throughput != null and .tg_throughput.mean != null) | if .tg_throughput.std > 0.5 then "\((.tg_throughput.mean * 10 + 0.5 | floor) / 10) +/-\((.tg_throughput.std * 10 + 0.5 | floor) / 10)" else "\((.tg_throughput.mean * 10 + 0.5 | floor) / 10)" end' "$local_json" | head -n1)
            [[ -z "${SUMMARY_TG[$MODEL]}" ]] && SUMMARY_TG[$MODEL]="—"

            SUMMARY_PEAK[$MODEL]=$(jq -r '.benchmarks[] | select(.context_size == 0) | select(.peak_throughput != null and .peak_throughput.mean != null) | .peak_throughput.mean + 0.5 | floor' "$local_json" | head -n1)
            [[ -z "${SUMMARY_PEAK[$MODEL]}" ]] && SUMMARY_PEAK[$MODEL]="—"

            SUMMARY_TTFT[$MODEL]=$(jq -r '.benchmarks[] | select(.context_size == 0) | select(.e2e_ttft != null and .e2e_ttft.mean != null) | .e2e_ttft.mean + 0.5 | floor' "$local_json" | head -n1)
            [[ -z "${SUMMARY_TTFT[$MODEL]}" ]] && SUMMARY_TTFT[$MODEL]="—"
            
            if [[ -n "$baseline_tg" && -n "$deepest_tg" && "$max_depth" -gt 0 ]]; then
                pct=$(echo "scale=1; (($deepest_tg - $baseline_tg) / $baseline_tg) * 100" | bc)
                SUMMARY_DEGRADATION[$MODEL]="${pct}% @$((max_depth/1024))k"
            fi
        fi

        # Quality pass (tool-eval-bench) on the same loaded model
        if [[ "$QUALITY" == true ]]; then
            run_quality "$MODEL" || true
            if [[ -n "$QUALITY_SCORE" ]]; then
                SUMMARY_QUALITY[$MODEL]="${QUALITY_SCORE}"
                [[ "$QUALITY_ONLY" == true ]] && SUMMARY_STATUS[$MODEL]="OK" && PASS=$((PASS + 1)) && checkpoint_model_done "$MODEL" "OK"
            else
                SUMMARY_QUALITY[$MODEL]="?"
                [[ "$QUALITY_ONLY" == true ]] && SUMMARY_STATUS[$MODEL]="FAIL" && FAIL=$((FAIL + 1)) && checkpoint_model_done "$MODEL" "FAIL"
            fi
            # Persist tool-eval-bench detail subscores for the final-report
            # quality table. Defaults to "—" so the table renders cleanly even
            # when a field couldn't be parsed.
            SUMMARY_QUALITY_DEPLOY[$MODEL]="${QUALITY_DEPLOY:-—}"
            SUMMARY_QUALITY_RESPONSE[$MODEL]="${QUALITY_RESPONSE_MS:-—}"
            SUMMARY_QUALITY_CTXPRESS[$MODEL]="${QUALITY_CTXPRESS:-—}"
            SUMMARY_QUALITY_POINTS[$MODEL]="${QUALITY_TOTAL_POINTS:-—}"
            SUMMARY_QUALITY_CATS[$MODEL]="${QUALITY_CATS:-—}"
            SUMMARY_QUALITY_REPORT[$MODEL]="${QUALITY_REPORT:-}"
            SUMMARY_QUALITY_RATING[$MODEL]="${QUALITY_RATING:-—}"
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
# Wider table when quality column is shown
if [[ "$QUALITY" == true ]]; then
    log "  ${BOLD}$(printf '%-42s  %8s  %14s  %12s  %8s  %14s  %9s' 'Model' 'Load' 'Read (pp)' 'Write (tg)' 'Peak' 'Deep ctx' 'Quality')${NC}"
    log "  ${DIM}$(printf '%-42s  %8s  %14s  %12s  %8s  %14s  %9s' '' 'sec' 'tok/s' 'tok/s' 'tok/s' 'degradation' '/100')${NC}"
    log "  $(printf '%.0s-' {1..120})"
else
    log "  ${BOLD}$(printf '%-42s  %8s  %14s  %12s  %8s  %14s' 'Model' 'Load' 'Read (pp)' 'Write (tg)' 'Peak' 'Deep ctx')${NC}"
    log "  ${DIM}$(printf '%-42s  %8s  %14s  %12s  %8s  %14s' '' 'sec' 'tok/s' 'tok/s' 'tok/s' 'degradation')${NC}"
    log "  $(printf '%.0s-' {1..106})"
fi

for MODEL in $MODELS; do
    local_name="$MODEL"
    [[ ${#local_name} -gt 42 ]] && local_name="${local_name:0:39}..."

    status="${SUMMARY_STATUS[$MODEL]:-FAIL}"
    pp="${SUMMARY_PP[$MODEL]:-—}"
    tg="${SUMMARY_TG[$MODEL]:-—}"
    peak="${SUMMARY_PEAK[$MODEL]:-—}"
    ttft="${SUMMARY_TTFT[$MODEL]:-—}"
    degrad="${SUMMARY_DEGRADATION[$MODEL]:-—}"
    quality="${SUMMARY_QUALITY[$MODEL]:-—}"

    if [[ "$status" == "OK" ]]; then
        if [[ "$QUALITY" == true ]]; then
            printf -v line "  %-42s  %8s  %14s  %12s  %8s  %14s  %9s" "$local_name" "${SUMMARY_LOAD[$MODEL]:-—}" "$pp" "$tg" "$peak" "$degrad" "$quality"
        else
            printf -v line "  %-42s  %8s  %14s  %12s  %8s  %14s" "$local_name" "${SUMMARY_LOAD[$MODEL]:-—}" "$pp" "$tg" "$peak" "$degrad"
        fi
        log "${GREEN}${line}${NC}"
    elif [[ "$status" == "INCOHERENT" ]]; then
        if [[ "$QUALITY" == true ]]; then
            printf -v line "  %-42s  %8s  %14s  %12s  %8s  %14s  %9s" "$local_name" "${SUMMARY_LOAD[$MODEL]:-—}" "INCOHERENT" "—" "—" "—" "—"
        else
            printf -v line "  %-42s  %8s  %14s  %12s  %8s  %14s" "$local_name" "${SUMMARY_LOAD[$MODEL]:-—}" "INCOHERENT" "—" "—" "—"
        fi
        log "${YELLOW}${line}${NC}"
    else
        if [[ "$QUALITY" == true ]]; then
            printf -v line "  %-42s  %8s  %14s  %12s  %8s  %14s  %9s" "$local_name" "—" "FAIL" "—" "—" "—" "—"
        else
            printf -v line "  %-42s  %8s  %14s  %12s  %8s  %14s" "$local_name" "—" "FAIL" "—" "—" "—"
        fi
        log "${RED}${line}${NC}"
    fi
done

log ""
log "  ${DIM}---------------------------------------------------------------${NC}"
log "  ${DIM}How to read this table:${NC}"
log "  ${DIM}${NC}"
log "  ${DIM}  Load         = Cold-load time (model load + first token reply).${NC}"
log "  ${DIM}  Read (pp)    = How fast the model reads your prompt (higher = better).${NC}"
log "  ${DIM}  Write (tg)   = How fast the model types its answer at depth=0 baseline.${NC}"
log "  ${DIM}                  This is the speed you feel when chatting.${NC}"
log "  ${DIM}                  Humans read at ~4 tok/s, so 20+ feels smooth.${NC}"
log "  ${DIM}  Peak         = Fastest burst speed observed in a 1-second window.${NC}"
log "  ${DIM}  Deep ctx     = Speed change at max tested depth vs baseline.${NC}"
log "  ${DIM}                  >-15% = unified memory bandwidth bottleneck.${NC}"
log "  ${DIM}  Quality      = tool-eval-bench overall score (0-100).${NC}"
log "  ${DIM}---------------------------------------------------------------${NC}"

# Tool-eval-bench detail block — only emitted when --quality (or --quality-only)
# was used and we have at least one parsed score. Shows the per-model
# subscores and category breakdown that the github page promises but the
# main table can't fit.
if [[ "$QUALITY" == true ]]; then
    log ""
    log "${BOLD}============================================================${NC}"
    log "${BOLD}  TOOL-EVAL-BENCH — quality detail${NC}"
    log "${BOLD}  github.com/SeraphimSerapis/tool-eval-bench${NC}"
    log "${BOLD}============================================================${NC}"
    log ""
    log "  ${BOLD}$(printf '%-42s  %7s  %7s  %8s  %10s  %8s  %s' 'Model' 'Score' 'Deploy' 'Median' 'CtxPress' 'Points' 'Rating')${NC}"
    log "  ${DIM}$(printf '%-42s  %7s  %7s  %8s  %10s  %8s  %s' '' '/100' '/100' 'turn (s)' '%' 'earned' '')${NC}"
    log "  $(printf '%.0s-' {1..120})"

    for MODEL in $MODELS; do
        local_name="$MODEL"
        [[ ${#local_name} -gt 42 ]] && local_name="${local_name:0:39}..."
        # Skip rows where the model didn't run quality at all
        if [[ -z "${SUMMARY_QUALITY[$MODEL]:-}" || "${SUMMARY_QUALITY[$MODEL]}" == "—" ]]; then
            continue
        fi
        printf -v line "  %-42s  %7s  %7s  %8s  %10s  %8s  %s" \
            "$local_name" \
            "${SUMMARY_QUALITY[$MODEL]:-—}" \
            "${SUMMARY_QUALITY_DEPLOY[$MODEL]:-—}" \
            "${SUMMARY_QUALITY_RESPONSE[$MODEL]:-—}" \
            "${SUMMARY_QUALITY_CTXPRESS[$MODEL]:-—}" \
            "${SUMMARY_QUALITY_POINTS[$MODEL]:-—}" \
            "${SUMMARY_QUALITY_RATING[$MODEL]:-—}"
        log "$line"
    done

    log ""
    # Per-model category breakdown — one line per model, only when we parsed it.
    have_cats=false
    for MODEL in $MODELS; do
        [[ -n "${SUMMARY_QUALITY_CATS[$MODEL]:-}" && "${SUMMARY_QUALITY_CATS[$MODEL]}" != "—" ]] && have_cats=true
    done
    if [[ "$have_cats" == true ]]; then
        log "  ${BOLD}Category scores per model:${NC}"
        for MODEL in $MODELS; do
            cats="${SUMMARY_QUALITY_CATS[$MODEL]:-}"
            [[ -z "$cats" || "$cats" == "—" ]] && continue
            log "  ${DIM}- ${MODEL}:${NC} $cats"
        done
        log ""
    fi

    log "  ${DIM}Field guide:${NC}"
    log "  ${DIM}  Score      = headline tool-call accuracy (earned / max points × 100).${NC}"
    log "  ${DIM}  Deploy     = Deployability subscore — combined fitness for production use.${NC}"
    log "  ${DIM}  Median turn= median seconds per assistant turn (lower = snappier).${NC}"
    log "  ${DIM}  CtxPress   = % of context window pre-filled before the test prompt.${NC}"
    log "  ${DIM}  Points     = raw points earned vs max for the chosen scenarios.${NC}"
    log "  ${DIM}  Rating     = star rating from tool-eval-bench (Excellent / Good / Fair / Poor).${NC}"
    log "  ${DIM}---------------------------------------------------------------${NC}"
fi
log ""
log "  Results saved to:"
log "    Report     : $REPORT_FILE"
log "    JSON data  : $RESULTS_DIR/*_${TIMESTAMP}.json"
log "    Forum tables: $RESULTS_DIR/*_${TIMESTAMP}.md"
[[ "$QUALITY" == true ]] && log "    Quality reports: $QUALITY_DIR/<run_id>/report.md"
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
