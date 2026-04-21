Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv 未安装，请先安装 uv。"
}

Write-Host "==> 仅构建 wheel（--wheel --no-sources）"
uv build --wheel --no-sources

Write-Host "==> 构建完成，wheel 产物位于 dist/"

