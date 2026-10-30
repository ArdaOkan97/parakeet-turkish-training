$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20'
$sampleScript = Join-Path $taskRoot 'outputs\parakeet_turkish\owned_dashboard_capacity_sample_v1.py'
$pythonRuntime = Join-Path $taskRoot 'work\training-env\Scripts\python.exe'
$releasePath = Join-Path $taskRoot 'outputs\parakeet_turkish\ROOT_owned_trackio19568_actual_capacity_release_v1.json'
$lease = Get-Content -LiteralPath (Join-Path $taskRoot 'work\gpu_training_owner.json') -Raw | ConvertFrom-Json
if ($lease.actual_GPU_running -or $lease.actual_completed_step -ne 34651 -or $lease.owner -ne '/root/benchmark_design' -or $lease.actual_controller_session_id -ne 99543) { throw 'Known native closure/lease differs.' }
$listener = @(Get-NetTCPConnection -LocalPort 7860 -State Listen)
if ($listener.Count -ne 1 -or $listener[0].OwningProcess -ne 19568 -or $listener[0].LocalAddress -ne '127.0.0.1') { throw 'Owned listener identity differs.' }
$identity = Get-CimInstance Win32_Process -Filter 'ProcessId=19568'
if ($identity.ExecutablePath -ne 'C:\Users\ardao\Miniconda3\envs\vlm\python.exe') { throw 'Server executable differs.' }
$expectedCreation = [datetimeoffset]'2026-10-10T21:52:07.757049Z'
if ([math]::Abs(($identity.CreationDate.ToUniversalTime() - $expectedCreation.UtcDateTime).TotalSeconds) -gt 1) { throw 'Server PID was reused.' }
$body = @{ kwargs = @{ project = 'parakeet-turkish' } } | ConvertTo-Json -Depth 4 -Compress
$project = Invoke-RestMethod -Uri 'http://127.0.0.1:7860/api/get_runs_for_project' -Method Post -ContentType 'application/json' -Body $body
if (-not ($project.data | Where-Object { $_.id -eq '6a6980b07df0400a8ec9e5eee066711c' -and $_.name -eq 'adaptive_20261012_TR_uniform_encoder_cosine_01' })) { throw 'Workspace project/run identity differs.' }
$server = Get-Process -Id 19568
$null = $server.Handle
$privateBefore = $server.PrivateMemorySize64
$baselineBefore = (& $pythonRuntime -X utf8 -S $sampleScript | ConvertFrom-Json)
$estimatedAfter = $baselineBefore.commit_used_bytes - $privateBefore + 12079239168
if ($estimatedAfter -ge .9 * $baselineBefore.commit_limit_bytes) { throw 'This scoped release would not plausibly repair admission; server left running.' }
Stop-Process -Id 19568 -Confirm:$false
if (-not $server.WaitForExit(10000)) { throw 'Actual server closure not observed.' }
$actualClosed = $server.HasExited
$serverExit = $server.ExitCode
$remaining = @(Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue)
$samples = @()
for ($index = 0; $index -lt 3; $index++) {
    Start-Sleep -Seconds 5
    $samples += (& $pythonRuntime -X utf8 -S $sampleScript | ConvertFrom-Json)
}
$release = [ordered]@{
    kind = 'ROOT_owned_trackio19568_actual_capacity_release_v1'
    scope = 'Only the specifically verified optional workspace Trackio dashboard server.'
    process_id = 19568
    process_creation_utc = $identity.CreationDate.ToUniversalTime().ToString('o')
    private_bytes_before = $privateBefore
    closure_method = 'Scoped Windows Stop-Process; not a training-worker stop or user application closure.'
    actual_owned_process_closed = $actualClosed
    actual_process_exit_code = $serverExit
    actual_listener_closed = ($remaining.Count -eq 0)
    baseline_before = $baselineBefore
    baseline_after = $samples[-1]
    after_samples = $samples
    user_apps_closed = $false
    pagefile_changed = $false
    guards_changed = $false
    database_or_historical_metrics_deleted = $false
    SDK_or_replacement_logger_started = $false
}
$release | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $releasePath -Encoding utf8
& $pythonRuntime -X utf8 -S $sampleScript $releasePath
exit $LASTEXITCODE
