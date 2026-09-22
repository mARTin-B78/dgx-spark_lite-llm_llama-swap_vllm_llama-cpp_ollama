#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
revision=76d51ef82a81b70b78e51a3a6ea11946286de976
if [[ ! -d ds4-0731/.git ]]; then
  git clone --no-checkout https://github.com/Entrpi/ds4.git ds4-0731
  git -C ds4-0731 checkout --detach "$revision"
fi
[[ "$(git -C ds4-0731 rev-parse HEAD)" == "$revision" ]] || {
  echo 'Existing ds4-0731 checkout has a different revision; refusing to replace it.' >&2; exit 1;
}
python3 setup/download-deepseek-0731.py
make -C ds4-0731 cuda-spark -j2
cp ds4-0731/ds4-server ds4-0731-container/ds4-server
# Base was pinned from the existing local CUDA 13 / Ubuntu 24.04 image.
docker image inspect local/ds4-runtime-base:cd5b279db639 >/dev/null
docker build -t local/ds4-0731:76d51ef ds4-0731-container
