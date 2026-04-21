#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv 未安装，请先安装 uv。" >&2
  exit 1
fi

echo "==> 仅构建 wheel（--wheel --no-sources）"
uv build --wheel --no-sources

echo "==> 构建完成，wheel 产物位于 dist/"

