$ErrorActionPreference='Stop'
$dataStorage=Get-Content -LiteralPath 'E:\gpu_training_scratchpad\storage_config.json' -Raw | ConvertFrom-Json
if($dataStorage.storage_root -ne 'E:\gpu_training_scratchpad' -or $dataStorage.allow_fallback_outside_root){throw 'E storage policy differs'}
. 'E:\gpu_training_scratchpad\use_storage.ps1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:TOKENIZERS_PARALLELISM='false'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
& 'E:\gpu_training_scratchpad\scratch\CPU_validation_v2_env\Scripts\python.exe' -X utf8 -B (Join-Path $PSScriptRoot 'refresh_ASR18_data_dispatch_v2.py')
exit $LASTEXITCODE
