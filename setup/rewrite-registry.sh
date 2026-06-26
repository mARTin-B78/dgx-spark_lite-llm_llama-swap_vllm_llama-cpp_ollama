#!/bin/bash
###############################################################################
# rewrite-registry.sh
#
# Rewrites every `ghcr.io/<NAMESPACE>/<image>` reference in the runtime
# config files to a different registry + namespace. Use this when you build
# the stack images yourself and push them to a private registry (GitLab,
# Harbor, Nexus, an internal ghcr.io org, …) instead of `ghcr.io/martin-b78`.
#
# The build/compose paths already honour ${REGISTRY} + ${IMAGE_NAMESPACE} from
# .env, but llama-swap's config.yaml inlines image names inside `docker run`
# command strings — env substitution does not happen there, so the strings
# have to be rewritten on disk. That's what this script does.
#
# Usage:
#   ./setup/rewrite-registry.sh                # interactive: read from .env
#   ./setup/rewrite-registry.sh --dry-run      # show what would change
#   REGISTRY=registry.gitlab.com \
#   IMAGE_NAMESPACE=mygroup/spark-stack \
#     ./setup/rewrite-registry.sh
#
# Files touched (each gets a .bak.<timestamp> backup):
#   llama-swap/config.yaml
#   llama-swap/config.yaml.sample
#
# Files left alone:
#   docker-compose.yml(.sample) — already use ${REGISTRY}/${IMAGE_NAMESPACE}
#   build_and_push.sh           — pushes to ${REGISTRY}/${IMAGE_NAMESPACE}
#   README.md / TUTORIAL.md     — examples; unchanged so the docs still
#                                  describe the upstream defaults.
###############################################################################

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Load .env if present so REGISTRY / IMAGE_NAMESPACE can come from there.
if [ -f "$REPO_ROOT/.env" ]; then
    # shellcheck disable=SC2046
    export $(grep -v '^#' "$REPO_ROOT/.env" | xargs -d '\n' -I{} echo {})
fi

REGISTRY="${REGISTRY:-}"
IMAGE_NAMESPACE="${IMAGE_NAMESPACE:-${REGISTRY_USER:-}}"

# What we're rewriting from. The repo's published images live here:
OLD_REGISTRY="ghcr.io"
OLD_NAMESPACE="martin-b78"

if [ -z "$REGISTRY" ] || [ -z "$IMAGE_NAMESPACE" ]; then
    echo "❌ Set REGISTRY and IMAGE_NAMESPACE first."
    echo "   Either:"
    echo "     1. Run ./setup/setup.sh and answer the registry questions, OR"
    echo "     2. Add REGISTRY=… and IMAGE_NAMESPACE=… to .env, OR"
    echo "     3. Pass them inline:  REGISTRY=registry.gitlab.com \\"
    echo "                            IMAGE_NAMESPACE=mygroup/spark-stack \\"
    echo "                            $0"
    exit 1
fi

if [ "$REGISTRY/$IMAGE_NAMESPACE" = "$OLD_REGISTRY/$OLD_NAMESPACE" ]; then
    echo "ℹ️  REGISTRY/IMAGE_NAMESPACE already matches the upstream default."
    echo "    Nothing to rewrite."
    exit 0
fi

OLD="$OLD_REGISTRY/$OLD_NAMESPACE"
NEW="$REGISTRY/$IMAGE_NAMESPACE"

FILES=(
    "llama-swap/config.yaml"
    "llama-swap/config.yaml.sample"
)

echo "🔄 Rewriting image references"
echo "   FROM:  $OLD"
echo "   TO:    $NEW"
echo ""

CHANGED=0
for f in "${FILES[@]}"; do
    [ -f "$REPO_ROOT/$f" ] || { echo "  ⏭  $f  (missing — skipping)"; continue; }

    # Count matches (excluding lines we'd skip)
    matches=$(grep -c "$OLD" "$REPO_ROOT/$f" || true)
    if [ "$matches" -eq 0 ]; then
        echo "  ✓ $f  (no references)"
        continue
    fi

    if $DRY_RUN; then
        echo "  → $f  ($matches references) — would update"
        grep -n "$OLD" "$REPO_ROOT/$f" | sed 's/^/      /'
    else
        ts=$(date +%Y%m%d_%H%M%S)
        cp "$REPO_ROOT/$f" "$REPO_ROOT/$f.bak.$ts"
        # Use | as sed delimiter since paths contain /
        sed -i "s|$OLD|$NEW|g" "$REPO_ROOT/$f"
        echo "  ✏️  $f  ($matches references rewritten, backup: $f.bak.$ts)"
    fi
    CHANGED=$((CHANGED + 1))
done

echo ""
if $DRY_RUN; then
    echo "Dry-run complete. Re-run without --dry-run to apply."
else
    if [ "$CHANGED" -gt 0 ]; then
        echo "✅ Rewrote $CHANGED file(s). Restart llama-swap so the new config takes effect:"
        echo "      docker restart llama-swap"
    else
        echo "ℹ️  No files needed updating."
    fi
fi
