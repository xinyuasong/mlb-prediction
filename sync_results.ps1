Set-Location -LiteralPath $PSScriptRoot
$log = Join-Path $PSScriptRoot "scheduler.log"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
Add-Content -LiteralPath $log -Value "================ $stamp GIT SYNC ================"
foreach ($f in @("slate_history.csv","backtest_results/metrics.json","backtest_results/calibration.png")) {
  if (Test-Path -LiteralPath $f) { git add -- $f 2>&1 | Out-Null }
}
git diff --cached --quiet
if ($LASTEXITCODE -eq 0) { Add-Content -LiteralPath $log -Value "nothing new to commit"; exit 0 }
$today = Get-Date -Format "yyyy-MM-dd"
git commit -m "Daily run $today" 2>&1 | ForEach-Object { Add-Content -LiteralPath $log -Value $_ }
git pull --rebase --autostash origin main 2>&1 | ForEach-Object { Add-Content -LiteralPath $log -Value $_ }
git push origin main 2>&1 | ForEach-Object { Add-Content -LiteralPath $log -Value $_ }
Add-Content -LiteralPath $log -Value "sync done"
exit 0
