$ErrorActionPreference = 'Stop'
$asrScratch = 'E:\gpu_training_scratchpad'
$asrFolder = Join-Path $asrScratch 'analysis\BPE9223_training_launch_v1\resident_RAM_original_interval_post140236_failure_actual_v1'
Get-Content -LiteralPath (Join-Path $asrScratch 'storage_config.json') -Raw | ConvertFrom-Json | Out-Null
. (Join-Path $asrScratch 'use_storage.ps1')
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
function Get-ASRRecord([string]$asrPath) {
 $asrFile = Get-Item -LiteralPath $asrPath
 return [ordered]@{path=$asrFile.FullName;bytes=$asrFile.Length;sha256=(Get-FileHash -LiteralPath $asrPath -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Assert-ASRPin($asrPin) {
 $asrActual = Get-ASRRecord ([string]$asrPin.path)
 if ($asrActual.bytes -ne $asrPin.bytes -or $asrActual.sha256 -cne $asrPin.sha256) { throw ('Exact pin mismatch: '+$asrPin.path) }
 return $asrActual
}
$asrScopePath = Join-Path $asrScratch 'analysis\BPE9223_training_launch_v1\ROOT_sixth_terminal_interval_CPU_fixture_source_scope_review_v1.json'
$asrScopePin = Assert-ASRPin ([pscustomobject]@{path=$asrScopePath;bytes=6220;sha256='a6d14fc2ee4139ccadcb4fbf39d22487dd918d373e368bc4f5eddec07b9cdda8'})
$asrScope = Get-Content -LiteralPath $asrScopePath -Raw | ConvertFrom-Json
$asrPins = @($asrScope.verified_source_pins.PSObject.Properties | ForEach-Object { $_.Value }) + @($asrScope.required_exact_input_pins_before_launch_and_after_completion) + @($asrScope.ROOT_source_gate_argument)
$asrBefore = @($asrPins | ForEach-Object { Assert-ASRPin $_ })
if (Test-Path -LiteralPath (Join-Path $asrFolder 'ingress_frontend_actual_01')) { throw 'Fresh actual output already exists' }
$asrLaunchPath = Join-Path $asrFolder 'ingress_frontend_actual_01_launch.json'
$asrWaitPath = Join-Path $asrFolder 'ingress_frontend_actual_01_external_wait.json'
if ((Test-Path -LiteralPath $asrLaunchPath) -or (Test-Path -LiteralPath $asrWaitPath)) { throw 'One approved attempt only' }
$asrPython = Join-Path $asrScratch 'scratch\BPE_broad_CPU_preflight_env_v2\Scripts\python.exe'
$asrArgs = @('-X','utf8','-B',(Join-Path $asrFolder 'verify_sixth_terminal_interval_ingress_frontend_v1.py'),'--ROOT-review',[string]$asrScope.ROOT_source_gate_argument.path)
$asrStartedUTC = [DateTime]::UtcNow.ToString('o')
$asrJob = Start-Process -FilePath $asrPython -ArgumentList $asrArgs -WorkingDirectory 'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20' -WindowStyle Hidden -RedirectStandardOutput (Join-Path $asrFolder 'ingress_frontend_actual_01.stdout.log') -RedirectStandardError (Join-Path $asrFolder 'ingress_frontend_actual_01.stderr.log') -PassThru
$asrJob.PriorityClass = [Diagnostics.ProcessPriorityClass]::BelowNormal
$asrCIM = Get-CimInstance Win32_Process -Filter ('ProcessId='+$asrJob.Id)
$asrLaunch = [ordered]@{kind='ASR18_sixth_ingress_CPU_actual_launch_v1';started_UTC=$asrStartedUTC;scope=$asrScopePin;input_pins_before=$asrBefore;launcher_pid=$asrJob.Id;launcher_creation_UTC=$asrCIM.CreationDate.ToUniversalTime().ToString('o');launcher_ppid=$asrCIM.ParentProcessId;argv=$asrArgs;storage_config=(Get-ASRRecord (Join-Path $asrScratch 'storage_config.json'));storage_environment_script=(Get-ASRRecord (Join-Path $asrScratch 'use_storage.ps1'));TORCH_EXTENSIONS_DIR=$env:TORCH_EXTENSIONS_DIR;CUDA_VISIBLE_DEVICES=$env:CUDA_VISIBLE_DEVICES;all_four_thread_caps=1;priority='BelowNormal'}
$asrLaunch | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $asrLaunchPath -Encoding UTF8
$asrLaunch | ConvertTo-Json -Depth 20 -Compress
$asrJob.WaitForExit()
$asrJob.Refresh()
$asrCode = $asrJob.ExitCode
$asrAfter = @($asrPins | ForEach-Object { Assert-ASRPin $_ })
[ordered]@{kind='ASR18_sixth_ingress_CPU_external_wait_v1';completed_UTC=[DateTime]::UtcNow.ToString('o');actual_Process_wait_exit_code=$asrCode;launch=(Get-ASRRecord $asrLaunchPath);input_pins_after=$asrAfter;all_exact_input_pins_unchanged=$true;launcher_absent=(Get-CimInstance Win32_Process -Filter ('ProcessId='+$asrJob.Id) | Measure-Object).Count -eq 0} | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $asrWaitPath -Encoding UTF8
Get-Content -LiteralPath $asrWaitPath -Raw
exit $asrCode
