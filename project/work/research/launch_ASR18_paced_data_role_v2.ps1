param([Parameter(Mandatory=$true)][ValidateSet('acquisition','archive_indexer')][string]$Role)
$ErrorActionPreference='Stop'
$dataRoot='E:\gpu_training_scratchpad'
$dataArea=Join-Path $dataRoot 'analysis\BPE_weak_then_clean_v1'
$dataProject='C:\Users\ardao\Documents\Codex\2026-10-10\i-x20'
$dataStorage=Get-Content -LiteralPath (Join-Path $dataRoot 'storage_config.json') -Raw | ConvertFrom-Json
. (Join-Path $dataRoot 'use_storage.ps1')
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
$dataOld=Get-Content -LiteralPath (Join-Path $dataArea 'old_heavy_supported_gate_exit_v2.json') -Raw | ConvertFrom-Json
if ($dataOld.native_exit_observed -ne 75 -or $dataOld.indexer_native_exit_observed -ne 1) { throw 'Actual old native closure proof differs' }
foreach($dataOldPID in @(31428,33988)) {
 $dataProc=Get-CimInstance Win32_Process -Filter "ProcessId=$dataOldPID"
 if($dataProc -and ($dataProc.CommandLine -match 'prepare_BPE_weak_then_clean_data_v1.py|index_BPE_YODAS_archives_v1.py')) { throw 'Original heavy worker remains live' }
}
$dataPermit=Get-Content -LiteralPath (Join-Path $dataArea 'ROOT_data_IO_control_v2.json') -Raw | ConvertFrom-Json
$dataNow=[DateTimeOffset]::UtcNow
# ConvertFrom-Json can produce UTC DateTime objects. Preserve their timezone;
# Parse(object) would stringify without its offset and assume the local zone.
$dataUntil=[DateTimeOffset]$dataPermit.valid_until_utc
$dataPublished=[DateTimeOffset]$dataPermit.updated_at_utc
if($dataPermit.owner -ne '/root/memory_diagnosis' -or $dataPermit.phase -ne 'steady_safe' -or $dataPermit.allow_heavy_IO -ne $true -or $dataPermit.permitted_roles.Count -ne 1 -or $dataPermit.permitted_roles[0] -ne $Role -or $dataUntil -le $dataNow -or $dataPublished -gt $dataNow -or ($dataUntil-$dataPublished).TotalSeconds -gt 300) { throw 'No explicit current one-role steady-safe I/O permit' }
$dataPolicy=Get-Content -LiteralPath (Join-Path $dataArea 'paced_successors_preparation_policy_v2.json') -Raw | ConvertFrom-Json
$dataSourceName=if($Role -eq 'acquisition'){'prepare_BPE_paced_acquisition_v2.py'}else{'index_BPE_YODAS_archives_v2.py'}
foreach($dataName in @('ASR18_data_io_guard_v2.py',$dataSourceName)) {
 $dataPath=Join-Path $dataProject ('work\research\'+$dataName)
 if((Get-FileHash -LiteralPath $dataPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $dataPolicy.source_SHA256.$dataName) { throw 'Pinned successor source changed' }
}
& (Join-Path $dataProject 'work\training-env\Scripts\python.exe') -X utf8 -B (Join-Path $dataProject ('work\research\'+$dataSourceName))
exit $LASTEXITCODE
