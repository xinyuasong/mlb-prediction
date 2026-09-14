# One-time setup: registers three Windows Task Scheduler jobs for the MLB model.
#
#   Run from PowerShell in this folder:   .\setup_schedule.ps1
#   (If blocked: powershell -ExecutionPolicy Bypass -File .\setup_schedule.ps1)
#
# Schedule (adjust the times below if you are not in the Eastern time zone -
# the close snapshots are timed to land just before typical first pitches):
#   10:00  morning slate  - full pipeline, EV flags, webhook
#   12:45  close snapshot - locks closing lines for afternoon games
#   18:40  close snapshot - locks closing lines for night games
#
# Remove later with:  .\setup_schedule.ps1 -Remove

param([switch]$Remove)

$proj = $PSScriptRoot
$tasks = @(
    @{ Name = "MLB Model - Morning Slate";  Bat = "run_morning.bat";     Time = "10:00" },
    @{ Name = "MLB Model - Close (day)";    Bat = "run_close.bat";       Time = "12:45" },
    @{ Name = "MLB Model - Close (early)";  Bat = "run_close.bat";       Time = "18:10" },
    @{ Name = "MLB Model - Close (night)";  Bat = "run_close.bat";       Time = "18:40" },
    @{ Name = "MLB Model - Weekly Eval";    Bat = "run_weekly_eval.bat"; Time = "09:00"; Weekly = $true }
)

if ($Remove) {
    foreach ($t in $tasks) {
        Unregister-ScheduledTask -TaskName $t.Name -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Removed: $($t.Name)"
    }
    exit
}

# StartWhenAvailable: if the run was missed entirely, fire as soon as the
# machine is back. WakeToRun: wake the laptop from sleep at the scheduled
# time - essential for the close snapshots, whose data ceases to exist once
# games start. (Requires wake timers enabled in Power Options; works from
# sleep, not from full shutdown.)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

foreach ($t in $tasks) {
    $action = New-ScheduledTaskAction -Execute "$proj\$($t.Bat)" -WorkingDirectory $proj
    if ($t.Weekly) {
        $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At $t.Time
        $when = "Mondays at $($t.Time)"
    } else {
        $trigger = New-ScheduledTaskTrigger -Daily -At $t.Time
        $when = "daily at $($t.Time)"
    }
    Register-ScheduledTask -TaskName $t.Name -Action $action -Trigger $trigger `
        -Settings $settings -Description "MLB prediction model automation" -Force | Out-Null
    Write-Host "Registered: $($t.Name)  ($when)"
}

Write-Host ""
Write-Host "Done. Verify in Task Scheduler (taskschd.msc) or test one now with:"
Write-Host '  Start-ScheduledTask -TaskName "MLB Model - Morning Slate"'
Write-Host "Runs log to scheduler.log in this folder."
