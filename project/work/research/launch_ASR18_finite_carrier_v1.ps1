param([Parameter(Mandatory=$true)][string]$Plan)
$ErrorActionPreference='Stop'
$asrDataConfig=Get-Content -LiteralPath 'E:\gpu_training_scratchpad\storage_config.json' -Raw | ConvertFrom-Json
if($asrDataConfig.storage_root -ne 'E:\gpu_training_scratchpad' -or $asrDataConfig.allow_fallback_outside_root){throw 'Storage policy mismatch'}
$asrPlanFull=[IO.Path]::GetFullPath($Plan)
if(-not $asrPlanFull.StartsWith('E:\gpu_training_scratchpad\',[StringComparison]::OrdinalIgnoreCase)){throw 'Pinned plan must be on E'}
. 'E:\gpu_training_scratchpad\use_storage.ps1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
& 'E:\gpu_training_scratchpad\scratch\CPU_validation_v2_env\Scripts\python.exe' -X utf8 -B (Join-Path $PSScriptRoot 'run_ASR18_FLAC_carrier_batch_v8.py') --plan $asrPlanFull
exit $LASTEXITCODE
