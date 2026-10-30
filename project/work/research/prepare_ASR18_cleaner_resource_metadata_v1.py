"""Full cleaner scalar/target audit and finite HEADER-ONLY proposal. No media open."""
import collections
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, AREA, ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO

P = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(P / 'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json

OUT = AREA / 'cleaner_resource_accounting_v1'
SOURCE = P / 'work/data/TR_expanded_WorldSpeech_index_v1'
LABELS = ROOT / 'indexes/BPE_full_decoder_expanded_labels_v1'
GROUPS = ('fleurs', 'cv', 'tsc', 'worldspeech')
COL = struct.Struct('<IIdB')
QUOTAS = dict(fleurs=2037, cv=2963, tsc=2500, worldspeech=2500)
SEED = 'ASR18_cleaner_header_plan_20261017_v1'
SAMPLE_FIELDS = ('samples', 'num_samples', 'audio_samples', 'sample_count', 'native_samples', 'actual_PCM_frames', 'pcm_frames')


def checked_int(value, minimum=1):
    assert type(value) is int and value >= minimum
    return value


def top(heap, value, ordinal, row):
    item = (value, -ordinal, row)
    if len(heap) < 8:
        heapq.heappush(heap, item)
    elif item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def main():
    assert not OUT.exists()
    OUT.mkdir()
    guard = DataIOGuard('cleaner_resource_metadata_full352768', heavy=False, maximum_bytes=4_000_000_000)
    guard.wait()
    io = MetadataIO(guard)
    source_receipt = io.json(SOURCE / 'index_provenance.json')
    label_receipt = io.json(LABELS / 'label_catalog_provenance.json')
    preparation = io.json(LABELS / 'actual_preparation_receipt.json')
    historical = io.json(ROOT / 'analysis/ASR18_memory_diagnosis_v1/cleaner_metadata_actual_01/report.json')
    assert source_receipt['rows'] == label_receipt['rows'] == 352768
    assert label_receipt['candidate_target_tokens'] == historical['identity']['full_target_tokens'] == 11972886
    assert io.binding(SOURCE / 'index_provenance.json')['sha256'] == '11c7aaa24f108876a53f0a53e1bed45fabb0d0781486435e1be37a2f862c3be9'
    assert io.binding(LABELS / 'label_catalog_provenance.json')['sha256'] == 'cc41d52a5f0787dc4e425664a252554542a30de8c73278ee9dde658e609353e8'
    assert preparation['passed'] is True and preparation['rows'] == 352768
    for name in ('offsets.u64', 'columns.bin'):
        actual = io.binding(SOURCE / name)
        assert all(actual[k] == v for k, v in source_receipt['files'][name].items())
    label_pin = io.binding(LABELS / 'labels.sqlite3')
    assert all(label_pin[k] == v for k, v in label_receipt['data'].items())
    src = sqlite3.connect((LABELS / 'labels.sqlite3').as_uri() + '?mode=ro&immutable=1', uri=True)
    src.execute('PRAGMA cache_size=-2048')
    cursor = src.execute('SELECT ordinal,source_id,frames,old_u,new_u,length(label_ids),new_labels_sha FROM labels ORDER BY ordinal')
    index = OUT / 'cleaner_resource_metadata_index_v1.sqlite'
    db = sqlite3.connect(index)
    db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA temp_store=MEMORY')
    db.execute('CREATE TABLE rows(ordinal INTEGER PRIMARY KEY,source_group TEXT,source_id TEXT UNIQUE,audio_filepath TEXT,duration REAL,declared_sr INTEGER,stored_T INTEGER,new_U INTEGER,full_target_SHA256 TEXT,exact_native_samples INTEGER,exact_channels INTEGER,source_declared_sr INTEGER,source_declared_channels INTEGER,recorded_WAV_size INTEGER,recorded_WAV_mtime_ns INTEGER)')
    counts = collections.Counter(); rates = collections.Counter(); source_rates = collections.Counter(); source_channels = collections.Counter(); fields = collections.Counter()
    sample_candidates = collections.Counter(); metrics = {name: [] for name in ('duration_metadata', 'stored_T', 'BPE_U', 'stored_T_times_U_plus1', 'dense_joint_BPE_bytes')}
    per_group = {name: dict(rows=0,tokens=0,MAX_duration_metadata=0,MAX_stored_T=0,MAX_U=0,exact_native_samples_unknown=0,exact_canonical_channels_unknown=0) for name in GROUPS}
    selections = {name: [] for name in GROUPS}
    row_sha = hashlib.sha256(); ids_sha = hashlib.sha256(); tokens = 0; ordinal = 0; offset = 0; duration_parts=[]; seconds=0.0
    row_file = SOURCE / 'rows.jsonl'
    before = row_file.stat()
    assert before.st_size == source_receipt['files']['rows.jsonl']['bytes']
    with row_file.open('rb') as rows, (SOURCE / 'columns.bin').open('rb') as cols, (SOURCE / 'offsets.u64').open('rb') as offs:
        assert struct.unpack('<Q',offs.read(8))[0] == 0
        while True:
            line = rows.readline(64*1024)
            if not line: break
            assert line.endswith(b'\n') and len(line)<64*1024
            row_sha.update(line); offset += len(line); ordinal += 1
            assert struct.unpack('<Q',offs.read(8))[0] == offset
            t,old_u,duration,group_id=COL.unpack(cols.read(COL.size)); group=GROUPS[group_id]
            row=json.loads(line); fields.update(row.keys())
            label=cursor.fetchone();assert label is not None
            assert label[0]==ordinal-1 and label[1]==row['source_id'] and label[2]==t==row['encoder_frames']
            assert label[3]==old_u==row['target_tokens']==len(row['tr_metadata_labels'])
            u=checked_int(label[4]);assert label[5]==2*u
            assert type(row['duration']) in (int,float) and math.isfinite(row['duration']) and row['duration']==duration>0
            sr=checked_int(row['sample_rate']);rates[sr]+=1
            sid=row['source_id'];ids_sha.update(sid.encode('utf-8')+b'\n')
            audio=row['audio_filepath'];allowed=any(audio.casefold().startswith(str(root).casefold()+'\\') for root in (P,ROOT))
            counts['outside_allowed_audio_path']+=int(not allowed)
            for field in SAMPLE_FIELDS:
                if field in row:sample_candidates[field]+=1
            original_sr=row.get('source_sample_rate');original_channels=row.get('source_channels')
            if type(original_sr) is int and original_sr>0:source_rates[original_sr]+=1
            if type(original_channels) is int and original_channels>0:source_channels[original_channels]+=1
            stat=row.get('tr_canonical_WAV_stat_at_preparation',{})
            counts['recorded_canonical_WAV_stat_rows']+=int(bool(stat))
            # No sample count is manufactured from seconds, encoder T, file size or source rate.
            db.execute('INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(ordinal,group,sid,audio,duration,sr,t,u,label[6],None,None,original_sr,original_channels,stat.get('size_bytes'),stat.get('mtime_ns')))
            detail=dict(ordinal=ordinal,source_group=group,source_id=sid,audio_filepath=audio,
                duration_metadata=duration,declared_canonical_sample_rate=sr,stored_encoder_frames=t,target_tokens=u,
                exact_native_samples=None,exact_canonical_channels=None,header_verified=False)
            values=dict(duration_metadata=duration,stored_T=t,BPE_U=u,stored_T_times_U_plus1=t*(u+1),dense_joint_BPE_bytes=t*(u+1)*8716*4)
            for metric,value in values.items():top(metrics[metric],value,ordinal,detail)
            priority=int.from_bytes(hashlib.sha256((SEED+'\0'+sid).encode('utf-8')).digest(),'big')
            selected=(-priority,-ordinal,detail)
            heap=selections[group]
            if len(heap)<QUOTAS[group]:heapq.heappush(heap,selected)
            elif selected[:2]>heap[0][:2]:heapq.heapreplace(heap,selected)
            g=per_group[group];g['rows']+=1;g['tokens']+=u;g['MAX_duration_metadata']=max(g['MAX_duration_metadata'],duration);g['MAX_stored_T']=max(g['MAX_stored_T'],t);g['MAX_U']=max(g['MAX_U'],u);g['exact_native_samples_unknown']+=1;g['exact_canonical_channels_unknown']+=1
            tokens+=u;duration_parts.append(duration)
            if ordinal%256==0:seconds+=math.fsum(duration_parts);duration_parts.clear()
            if ordinal%512==0:
                guard.pace(512*2100);db.commit()
                write_json(OUT/'status.json',dict(at_utc=now(),PID=os.getpid(),rows=ordinal,total_rows=352768,audio_files_opened=0,TRAIN_admitted=False))
        assert not cols.read(1) and not offs.read(1)
    seconds+=math.fsum(duration_parts)
    assert cursor.fetchone() is None;src.close();db.commit();db.close()
    assert ordinal==352768 and tokens==11972886 and row_sha.hexdigest()==source_receipt['files']['rows.jsonl']['sha256']
    assert counts['outside_allowed_audio_path']==0, 'Do not inspect any audio path outside authorized existing project/E roots'
    assert row_file.stat().st_size==before.st_size and row_file.stat().st_mtime_ns==before.st_mtime_ns
    assert abs(seconds/3600-source_receipt['hours'])<1e-8
    chosen=sorted((detail for heap in selections.values() for _,_,detail in heap),key=lambda v:v['ordinal'])
    assert len(chosen)==len({v['ordinal'] for v in chosen})==10000
    manifest=OUT/'header_only_first10000_manifest_v1.jsonl'
    with manifest.open('x',encoding='utf-8',newline='\n') as f:
        for item in chosen:f.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
    index_pin=io.binding(index);manifest_pin=io.binding(manifest)
    plan=dict(kind='ASR18_cleaner_finite_header_only_prepared_plan_v1',schema_version=1,owner='/root/cpu_data_preparation',
        at_utc=now(),prepared_only=True,execution_started=False,selection_seed=SEED,selection_quotas=QUOTAS,
        selected_rows=10000,total_population=352768,all_rows_retained=True,selection_manifest=manifest_pin,full_metadata_index=index_pin,
        maximum_files=10000,maximum_header_bytes_per_file=16384,maximum_RIFF_chunks_per_file=64,
        maximum_header_phase_IO_bytes=256000000,maximum_new_outputs_bytes=32000000,CPU_children=1,
        CPU_private_hard_cap_bytes=2000000000,incremental_private_forecast_bytes=64000000,forecast_margin_bytes=250000000,
        canonical_audio_only=True,no_original_archive_or_payload_scan=True,no_PCM_decode_or_write=True,
        parser_requirements=['RIFF/WAVE or explicit RF64 ds64; bounded fmt/data chunk headers only; seek past payload',
            'Validate rate/channels/format/block_align/data_size and exact divisible data frames; unsupported formats remain explicit unknown',
            'Bind source file size/mtime before/after plus header-byte hash; retain original WAV and all rows',
            'Compare full decoded canonical rate/channels/sample count/encoder T before consumed draw; stop/reconcile mismatches'],
        complete_population_header_coverage=False,no_production_admission=True)
    write_json(OUT/'header_only_plan_v1.json',plan)
    summary=dict(kind='ASR18_complete_cleaner_metadata_resource_accounting_v1',schema_version=1,owner='/root/cpu_data_preparation',
        at_utc=now(),PID=os.getpid(),complete_metadata_scan=True,rows=ordinal,full_target_tokens=tokens,population_mean_U=tokens/ordinal,
        hours_from_existing_duration_metadata=seconds/3600,augmentation_applied=False,augmentation_bypassed=True,
        exact_native_sample_rows=0,exact_native_sample_unknown_rows=ordinal,exact_canonical_channel_rows=0,exact_canonical_channel_unknown_rows=ordinal,
        exact_canonical_native_samples_MAX=None,exact_canonical_channels_MAX=None,
        declared_canonical_sample_rate_counts=dict(rates),explicit_original_source_rate_counts=dict(source_rates),explicit_original_source_channel_counts=dict(source_channels),
        explicit_original_source_rate_unknown_rows=ordinal-sum(source_rates.values()),explicit_original_source_channel_unknown_rows=ordinal-sum(source_channels.values()),
        unvalidated_named_sample_fields_not_promoted_to_canonical_header_evidence=True,
        per_source_group=per_group,counts=dict(counts),metadata_field_presence=dict(fields),explicit_sample_field_candidates=dict(sample_candidates),
        maxima={metric:[dict(value=value,**detail) for value,_,detail in sorted(heap,reverse=True)] for metric,heap in metrics.items()},
        bindings=dict(source_provenance=io.binding(SOURCE/'index_provenance.json'),source_rows=dict(path=str(row_file),bytes=before.st_size,mtime_ns=before.st_mtime_ns,sha256=row_sha.hexdigest()),
            BPE_provenance=io.binding(LABELS/'label_catalog_provenance.json'),BPE_actual_preparation=io.binding(LABELS/'actual_preparation_receipt.json'),BPE_database=label_pin,
            prior_actual_cleaner_metadata=io.binding(ROOT/'analysis/ASR18_memory_diagnosis_v1/cleaner_metadata_actual_01/report.json'),
            frozen_cleaner_rows=io.binding(P/'outputs/parakeet_turkish/bpe_cleaner_data_v1.py'),frozen_clean_phase_reader=io.binding(P/'outputs/parakeet_turkish/fastmss_clean_phase_reader_v1.py'),
            metadata_index=index_pin,header_only_plan=io.binding(OUT/'header_only_plan_v1.json'),producer_source=io.binding(__file__)),
        resource_proposal=dict(status='metadata_complete_exact_audio_preallocation_unknown_until_header_evidence',
            parallel_cleaner_decode_jobs=1,augmentation_transport_buffers=0,augmentation_output_buffers=0,
            waveform_arrays='Original cleaner reader output remains float32 mono16k; no additional augmentation copies. Full preallocation requires exact WAV header N/rate/channels and runtime match.',
            duration_or_T_or_file_size_converted_to_exact_samples=False,
            conditional_array_formula='4*N bytes per mono float32 waveform, plus actual proven reader lifetime/copies; N remains unknown here.',
            CPU_resource_guard_or_raw_pool_limit_changed=False,source_or_target_selection_added=False),
        audio_files_opened=0,audio_decoded=False,model_imported=False,GPU_used=False,
        quality_or_language_filter_added=False,full_PCM_or_native_resource_fit_claimed=False,production_enabled=False,TRAIN_admitted=False)
    io.unchanged();write_json(OUT/'report.json',summary)
    assert not any(name in sys.modules for name in ('torch','av','soundfile','numpy','scipy','pyarrow','transformers'))
    print(dict(cleaner_metadata_resource_report=io.binding(OUT/'report.json'),header_only_plan=io.binding(OUT/'header_only_plan_v1.json'),rows=ordinal,tokens=tokens,actual_owned_exit_still_required=True))


if __name__=='__main__':main()
