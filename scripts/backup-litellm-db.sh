#!/usr/bin/env bash
# Nightly pg_dump of litellm-postgres -> LiteLLM/backup/litellm-YYYYmmdd-HHMM.sql.gz (mode 600, keeps last 14).
# Dumps contain secrets (git-ignored: LiteLLM/backup/). Credentials are read from the container env.
# Restore (into an empty/replaced DB; stop litellm first):
#   zcat LiteLLM/backup/litellm-<stamp>.sql.gz | docker exec -i litellm-postgres sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
set -Eeuo pipefail
umask 077

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="$repo_root/LiteLLM/backup"
out="$dir/litellm-$(date +%Y%m%d-%H%M).sql.gz"
mkdir -p "$dir"

docker exec litellm-postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB"' | gzip > "$out.tmp"
[[ -s "$out.tmp" ]] && zcat "$out.tmp" | grep -q 'PostgreSQL database dump complete' \
    || { rm -f "$out.tmp"; echo "backup failed" >&2; exit 1; }
mv "$out.tmp" "$out"
chmod 600 "$out"

ls -1t "$dir"/litellm-*.sql.gz | tail -n +15 | xargs -r rm -f --
echo "$out"
