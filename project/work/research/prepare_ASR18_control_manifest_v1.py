"""CPU metadata-only exact diagnostic-control selection; no TRAIN pool edit."""
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import psutil

PROJECT=Path(__file__).resolve().parents[2]
ROOT=Path(r'E:\gpu_training_scratchpad')
AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1/control_manifest_v3'
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json
from resource_watchdog_v2 import ResourceGuard,sample_memory
from tsc_bpe_full_decoder_transition_v1 import preflight,record,checked
from expanded_training_rows_v1 import reopen_approved_expanded_TR_rows
from bpe_training_label_catalog_v1 import BPETrainingLabels

def guard():
    policy=ResourceGuard()
    while True:
        value=sample_memory(os.getpid(),'CPU_diagnostic_control_metadata')
        reason,_=policy.observe(value['system_memory'],value['worker_private_memory'])
        if reason or value['worker_private_memory']['private_bytes']>=2_000_000_000:
            write_json(AREA/'resource_stop.json',{'reason':reason or 'CPU_private2GB','sample':value});os._exit(75)
        time.sleep(1)

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    for key in ('TEMP','TMP','TORCH_EXTENSIONS_DIR'):assert Path(os.environ[key]).resolve().is_relative_to(ROOT.resolve())
    AREA.mkdir(parents=True,exist_ok=False);threading.Thread(target=guard,daemon=True).start()
    config=json.loads((ROOT/'runs/adaptive_20261015_TR_BPE512_full_decoder_02/run_config.json').read_text())
    context=preflight(SimpleNamespace(**config['arguments']))
    pool=json.loads(Path(checked(context['source_training_pool'])).read_text())
    tr=reopen_approved_expanded_TR_rows(Path(pool['turkish_index']['path']).parent,cache_size=256)
    labels=BPETrainingLabels(context['BPE_label_directory'],tr)
    labels.connection.execute('PRAGMA cache_size=-2048');labels.connection.execute('PRAGMA temp_store=MEMORY')
    candidates=labels.connection.execute('SELECT ordinal,source_id,full_joint_bytes FROM labels '
        'WHERE full_joint_bytes BETWEEN 8000000 AND 16000000 ORDER BY full_joint_bytes DESC LIMIT 14').fetchall()
    assert len(candidates)==14
    rows=[];ordinals=[];shapes=[]
    for ordinal,source_id,joint in candidates:
        row=tr[ordinal];assert row['source_id']==source_id
        rewritten=labels.rewrite_batch([row])[0]
        actual=rewritten['encoder_frames']*(rewritten['target_tokens']+1)*8716*4
        assert actual==joint
        rows.append(row);ordinals.append(ordinal)
    for language in ('pt','en'):
        row=min((r for r in pool['replay'] if r['language']==language),
                key=lambda r:r['encoder_frames']*(r['target_tokens']+1))
        rows.append(row)
    for row in rows:
        rewritten=labels.rewrite_batch([row])[0]
        joint=rewritten['encoder_frames']*(rewritten['target_tokens']+1)*8716*4
        print(json.dumps({'language':row['language'],'joint_bytes':joint,'frames':rewritten['encoder_frames'],
                          'tokens':rewritten['target_tokens'],'duration':row['duration']}),flush=True)
        assert joint>0
        shapes.append({'source_id':row['source_id'],'language':row['language'],'encoder_frames':rewritten['encoder_frames'],
                       'target_tokens':rewritten['target_tokens'],'full_joint_bytes':joint})
    assert len({row['source_id'] for row in rows})==16
    manifest={'kind':'ASR18_CPU_prevalidated_small_geometry_control_v1','rows':rows,'TR_ordinals':ordinals,
        'shapes':shapes,'maximum_control_joint_bytes':max(item['full_joint_bytes'] for item in shapes),
        'source_training_pool':context['source_training_pool'],'BPE_labels':labels.receipt['data'],
        'diagnostic_scope_only':True,'training_pool_cutoff_or_filter_introduced':False,
        'native_full_joint_guard_bytes_unchanged':1_700_000_000}
    path=AREA/'manifest.json';write_json(path,manifest)
    assert not any(name in sys.modules for name in ('torch','transformers','nemo','numba'))
    report={'passed':True,'GPU_used':False,'audio_decoded':False,'ML_runtime_imported':False,'manifest':record(path),
        'actual_rows':16,'TR_PT_EN':[14,1,1],'max_full_joint_bytes':manifest['maximum_control_joint_bytes'],
        'private_bytes':psutil.Process().memory_info().private,'scope':'diagnostic control; no production admission or quality filter'}
    write_json(AREA/'report.json',report);print(json.dumps(report),flush=True)
    labels.close();tr.close()

if __name__=='__main__':main()
