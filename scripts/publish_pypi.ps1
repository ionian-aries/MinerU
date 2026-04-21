Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv 未安装，请先安装 uv。"
}

if ([string]::IsNullOrWhiteSpace($env:UV_PUBLISH_TOKEN)) {
    throw "未检测到 UV_PUBLISH_TOKEN，请先设置 PyPI token。示例: `$env:UV_PUBLISH_TOKEN='pypi-xxxx'"
}

Write-Host "==> 发布到 PyPI"
$wheels = Get-ChildItem -Path "dist" -Filter "*.whl" -File
if ($wheels.Count -eq 0) {
    throw "未找到 dist/*.whl，请先执行 scripts/build.ps1。"
}

Write-Host "==> 仅发布 wheel 到 PyPI"
uv publish @($wheels.FullName)

Write-Host "==> 发布命令执行完成"

