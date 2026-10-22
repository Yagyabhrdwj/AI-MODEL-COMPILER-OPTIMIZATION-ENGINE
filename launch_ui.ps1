# AI Model Compiler - Web Dashboard Launcher
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host "  AI Model Compiler & Optimization Engine - Web Dashboard" -ForegroundColor White
Write-Host "===================================================================" -ForegroundColor Cyan

$python = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $python) {
    Write-Host "[INFO] Starting local server via Python..." -ForegroundColor Green
    Set-Location $root
    python server.py
} else {
    Write-Host "[INFO] Opening standalone web dashboard in browser..." -ForegroundColor Yellow
    Start-Process (Join-Path $root "frontend\index.html")
}
