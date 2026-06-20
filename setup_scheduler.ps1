<#
.SYNOPSIS
    Registers the Missed-Lead Detector to run daily via Windows Task Scheduler.
.DESCRIPTION
    Creates a scheduled task that runs the pipeline every day at 9:00 AM.
    Run this script as Administrator for best results.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1
#>

$taskName = "MissedLeadDetector"
$scriptPath = "$PSScriptRoot\run_daily.bat"
$taskPath = "\MissedLeadDetector"

Write-Host "============================================================"
Write-Host "  Setting up Missed-Lead Detector Daily Schedule"
Write-Host "============================================================"
Write-Host ""
Write-Host "  Task Name : $taskName"
Write-Host "  Script    : $scriptPath"
Write-Host "  Schedule  : Daily at 9:00 AM"
Write-Host ""

# Check if the batch file exists
if (-not (Test-Path $scriptPath)) {
    Write-Host "  [ERROR] Batch file not found: $scriptPath" -ForegroundColor Red
    exit 1
}

# Check if task already exists
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  [INFO] Task '$taskName' already exists. Removing and recreating..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

# Create the scheduled task action
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`""

# Create the trigger (daily at 9:00 AM)
$trigger = New-ScheduledTaskTrigger -Daily -At "09:00"

# Create the principal (run as current user)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

# Register the task
try {
    Register-ScheduledTask -TaskName $taskName `
                           -Action $action `
                           -Trigger $trigger `
                           -Principal $principal `
                           -Description "Missed-Lead Detector - daily check for missed customer inquiries" `
                           -Force

    Write-Host "  [OK] Task '$taskName' registered successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  The pipeline will run daily at 9:00 AM."
    Write-Host "  Logs will be saved to: $PSScriptRoot\logs\"
    Write-Host ""

    # Show the task info
    Get-ScheduledTask -TaskName $taskName | Format-List TaskName, State, Actions, Triggers
}
catch {
    Write-Host "  [ERROR] Failed to register task: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Try running PowerShell as Administrator and try again."
    exit 1
}

Write-Host "============================================================"
Write-Host "  To manually test: double-click run_daily.bat"
Write-Host "  To view in Task Scheduler: taskschd.msc"
Write-Host "  To run immediately: Start-ScheduledTask -TaskName '$taskName'"
Write-Host "============================================================"
