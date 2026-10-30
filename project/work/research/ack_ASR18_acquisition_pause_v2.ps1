$ErrorActionPreference='Stop'
$dataStorage=Get-Content -LiteralPath 'E:\gpu_training_scratchpad\storage_config.json' -Raw | ConvertFrom-Json
. 'E:\gpu_training_scratchpad\use_storage.ps1'
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
[Diagnostics.Process]::GetCurrentProcess().PriorityClass='BelowNormal'
$dataArea='E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1'
$dataBinding=Get-Content -LiteralPath (Join-Path $dataArea 'owned_paced_acquisition_dispatch_v2.json') -Raw | ConvertFrom-Json
$dataOwnedPID=[int]$dataBinding.PID
$dataControl=Get-Content -LiteralPath (Join-Path $dataArea 'ROOT_data_IO_control_v2.json') -Raw | ConvertFrom-Json
$dataState=Get-Content -LiteralPath (Join-Path $dataArea 'acquisition_status_v2.json') -Raw | ConvertFrom-Json
$dataAcquisitionGranted=($dataControl.allow_heavy_IO -eq $true -and $dataControl.phase -eq 'steady_safe' -and $dataControl.permitted_roles -contains 'acquisition')
if($dataAcquisitionGranted -or $dataControl.owner -ne '/root/memory_diagnosis' -or $dataState.pid -ne $dataOwnedPID -or $dataState.status -ne 'paused_ROOT_sensitive_phase' -or $dataState.partial_flushed -ne $true){throw 'Owned pause callback not observed'}
$dataPlan=Get-Content -LiteralPath (Join-Path $dataArea 'preparation_plan_v1.json') -Raw | ConvertFrom-Json
$dataJobs=@($dataPlan.download_jobs | Where-Object {$_.path -eq $dataState.current_file})
if($dataJobs.Count -ne 1){throw 'Current partial identity differs'}
$dataPartial=$dataJobs[0].destination+'.part'
function Get-DataSnapshot {
 $dataNative=Get-CimInstance Win32_Process -Filter "ProcessId=$dataOwnedPID"
 if(-not $dataNative -or $dataNative.CommandLine -notmatch 'prepare_BPE_paced_acquisition_v2.py'){throw 'Owned source PID differs'}
 $dataPartialItem=Get-Item -LiteralPath $dataPartial -ErrorAction SilentlyContinue
 return [ordered]@{utc=[DateTimeOffset]::UtcNow.ToString('o');partial_exists=($null -ne $dataPartialItem);partial_bytes=$(if($dataPartialItem){$dataPartialItem.Length}else{0});io=(Get-Content -LiteralPath (Join-Path $dataArea 'acquisition_IO_status_v2.json') -Raw | ConvertFrom-Json);read_bytes=[uint64]$dataNative.ReadTransferCount;write_bytes=[uint64]$dataNative.WriteTransferCount}
}
$dataBefore=Get-DataSnapshot
Start-Sleep -Seconds 3
$dataAfter=Get-DataSnapshot
$dataCurrentControl=Get-Content -LiteralPath (Join-Path $dataArea 'ROOT_data_IO_control_v2.json') -Raw | ConvertFrom-Json
$dataAcquisitionGranted=($dataCurrentControl.allow_heavy_IO -eq $true -and $dataCurrentControl.phase -eq 'steady_safe' -and $dataCurrentControl.permitted_roles -contains 'acquisition')
if($dataAcquisitionGranted -or $dataBefore.io.status -ne 'paused_no_heavy_IO' -or $dataAfter.io.status -ne 'paused_no_heavy_IO' -or $dataBefore.partial_exists -ne $dataAfter.partial_exists -or $dataBefore.partial_bytes -ne $dataAfter.partial_bytes -or $dataBefore.io.aggregate_paced_bytes -ne $dataAfter.io.aggregate_paced_bytes){throw 'Heavy pause is not stable'}
$dataAck=[ordered]@{kind='ASR18_actual_acquisition_pause_ACK_v2';at_utc=[DateTimeOffset]::UtcNow.ToString('o');owned_session=$dataBinding.session;owned_PID=$dataOwnedPID;process_alive=$true;status='paused_no_heavy_IO';callback_status=$dataState.status;partial_flushed=$true;HTTP_response_body_close_callback_executed=$true;idle_established_TCP_connections=@(Get-NetTCPConnection -OwningProcess $dataOwnedPID -State Established -ErrorAction SilentlyContinue).Count;TCP_zero_claimed=$false;stable_heavy_counter_interval_seconds=3;partial_path=$dataPartial;partial_bytes_before=$dataBefore.partial_bytes;partial_bytes_after=$dataAfter.partial_bytes;paced_bytes_before=$dataBefore.io.aggregate_paced_bytes;paced_bytes_after=$dataAfter.io.aggregate_paced_bytes;native_small_control_read_bytes_delta=($dataAfter.read_bytes-$dataBefore.read_bytes);native_write_bytes_delta=($dataAfter.write_bytes-$dataBefore.write_bytes);heavy_counters_stable=$true;small_control_polling_and_heartbeat_continue=$true;source_pin=$dataJobs[0].revision;source_or_environment_edited=$false;control=$dataCurrentControl;TRAIN_admitted=$false;heavy_successor_remains_resumable=$true}
$dataAckPath=Join-Path $dataArea 'DATA_heavy_pause_ACK_v2.json'
$dataLedger=Join-Path $dataArea 'DATA_heavy_pause_ACK_ledger_v2.jsonl'
if(-not (Test-Path -LiteralPath $dataLedger) -and (Test-Path -LiteralPath $dataAckPath)){
 $dataOldAck=Get-Content -LiteralPath $dataAckPath -Raw | ConvertFrom-Json
 $dataOldAck | ConvertTo-Json -Depth 8 -Compress | Add-Content -LiteralPath $dataLedger -Encoding utf8
}
$dataAck | ConvertTo-Json -Depth 8 -Compress | Add-Content -LiteralPath $dataLedger -Encoding utf8
$dataAck | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath ($dataAckPath+'.writing') -Encoding utf8
Move-Item -LiteralPath ($dataAckPath+'.writing') -Destination $dataAckPath -Force
[pscustomobject]$dataAck | Select-Object at_utc,owned_session,owned_PID,partial_bytes_after,paced_bytes_after,native_small_control_read_bytes_delta,native_write_bytes_delta,idle_established_TCP_connections,heavy_counters_stable | ConvertTo-Json
