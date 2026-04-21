#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv 未安装，请先安装 uv。" >&2
  exit 1
fi

if [ -z "${UV_PUBLISH_TOKEN:-}" ]; then
  echo "Error: 未检测到 UV_PUBLISH_TOKEN，请先导出 PyPI token。" >&2
  echo '示例: export UV_PUBLISH_TOKEN="pypi-xxxx"' >&2
  exit 1
fi

if ! ls dist/*.whl >/dev/null 2>&1; then
  echo "Error: 未找到 dist/*.whl，请先执行 bash scripts/build.sh。" >&2
  exit 1
fi

echo "==> 仅发布 wheel 到 PyPI"
uv publish dist/*.whl

echo "==> 发布命令执行完成"

