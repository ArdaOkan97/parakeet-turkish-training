"""Actual historical saved-format compatibility, never diagnostic admission."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/'outputs/parakeet_turkish';sys.path.insert(0,str(SOURCE))
import asr18_full_run_serial_DEV_v2 as dev
from fastmss_row_serial_cpu_v4 import environment,resource_guard

ACTUAL=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1\feature_window_native20_01\after_update20.state.json')


def main(output):
    environment();resource_guard();output=dev.new_E(output)
    dev.require(not output.exists(),'Fresh E-only saved-format controls');output.mkdir()
    checks=[]
    def check(name,value):dev.require(value,name);checks.append(name)
    def reject(name,metadata):
        try:dev.checkpoint_format_view(metadata)
        except (ValueError,KeyError,TypeError):checks.append(name);return
        raise AssertionError('Expected saved-format rejection: '+name)
    source=SOURCE/'asr18_full_run_serial_DEV_v2.py'
    ast.parse(source.read_text(encoding='utf-8'))
    before=dev.record(ACTUAL);metadata=dev.read_bound(before)
    global_rng=random.getstate();proof=dev.checkpoint_format_view(metadata)
    check('Actual frozen writer uses payload and has no raw_payload key','payload' in metadata and 'raw_payload' not in metadata)
    check('Unmodified real diagnostic20 format passes all saved-field paths',proof['passed'] and proof['optimizer_updates']==20)
    check('Historical diagnostic identity remains diagnostic; no fullrun admission',proof['diagnostic_only'] is True and proof['full_run_checkpoint_admitted'] is False)
    check('Actual named207 weight references and four RNG encodings checked',proof['weight_count']==207 and proof['all4_RNG_field_paths_format_checked'])
    check('Actual clocks preserve three original clocks plus20',proof['saved_source_clocks']==[64270,2020,62270])
    check('Format parsing does not consume global Python RNG',random.getstate()==global_rng)
    check('Format parsing loads no tensor values',proof['tensor_values_loaded_or_compared'] is False)
    # This is a rejected diagnostic pointer, not fabricated native/fullrun proof.
    pointer={'kind':'ASR18_production_window_completed_checkpoint_pointer_v2','diagnostic_only':True,
        'writer_returned_and_payload_verified':True,'all_window_updates_committed':True,'committed_updates':20,
        'checkpoint':before,'payload':metadata['payload']}
    pin=dev.write_new(output/'explicit_rejected_diagnostic_pointer.json',pointer)
    try:dev.checked_checkpoint(pin,identity={},expected_updates=20,boundary={})
    except ValueError:checks.append('Format-compatible actual diagnostic checkpoint still blocked by fullrun gate')
    else:raise AssertionError('Diagnostic checkpoint was admitted')
    def clone():return deepcopy(metadata)
    def nodes(m):return dev._saved_dictionary(m['tree'])
    def rename(m,old,new):
        for pair in m['tree'][1]:
            if pair[0][1]==old:pair[0][1]=new;return
        raise AssertionError('Required original key absent')
    bad=clone();bad['raw_payload']=bad.pop('payload');reject('Wrong raw_payload alias rejected',bad)
    for key in nodes(metadata):
        bad=clone();bad['tree'][1]=[pair for pair in bad['tree'][1] if pair[0][1]!=key]
        reject('Missing exact saved field '+key+' rejected',bad)
    for key in ('python_rng','numpy_rng','torch_rng','cuda_rng'):
        bad=clone();rename(bad,key,key+'_wrong_name');reject('Wrong '+key+' field path rejected',bad)
    bad=clone();nodes(bad)['numpy_rng'][0]='list';reject('NumPy tuple/list type drift rejected',bad)
    bad=clone();nodes(bad)['python_rng'][1][0]=['scalar',-1];reject('Invalid actual Python RNG encoding rejected',bad)
    bad=clone();idx=nodes(bad)['torch_rng'][1];bad['tensors'][idx]['dtype']='float32';reject('Torch RNG tensor dtype drift rejected',bad)
    bad=clone();nodes(bad)['torch_rng'][1]=True;reject('Boolean Torch RNG reference rejected',bad)
    bad=clone();nodes(bad)['cuda_rng'][0]='tuple';reject('CUDA RNG saved container type drift rejected',bad)
    bad=clone();nodes(bad)['weights'][1].pop();reject('Missing one named207 weight rejected',bad)
    bad=clone();pairs=nodes(bad)['weights'][1];pairs[1][1]=deepcopy(pairs[0][1]);reject('Duplicate weight tensor alias rejected',bad)
    bad=clone();dev.select(bad['tree'],'optimizer','updates')[1]=True;reject('Boolean saved optimizer frontier rejected',bad)
    bad=clone();dev.select(bad['tree'],'optimizer','clocks')[1][0][1]+=1;reject('Wrong saved source clock rejected',bad)
    bad=clone();bad['source']['sha256']='0'*64;reject('Changed writer source binding rejected',bad)
    bad=clone();bad['payload']['bytes']-=1;reject('Payload/tensor extent mismatch rejected',bad)
    bad=clone();prefix=dev.select(bad['tree'],'augmentation','committed_journal');prefix[1]=[pair for pair in prefix[1] if pair[0][1]!='offset'];reject('Missing authoritative journal prefix offset rejected',bad)
    check('Actual existing checkpoint metadata untouched',dev.record(ACTUAL)==before)
    check('No numerical/audio/model runtime imported',not any(k in sys.modules for k in ('numpy','torch','scipy','transformers','pyarrow','soundfile','av')))
    check('No tensor payload loaded or rewritten',not any(p.suffix=='.bin' for p in output.iterdir()))
    resource_guard()
    report={'kind':'ASR18_full_run_serial_DEV_actual_saved_format_controls_v1','passed':True,
        'checks':checks,'check_count':len(checks),'source':dev.record(__file__),'implementation':dev.record(source),
        'actual_historical_checkpoint_metadata_read_only':before,'actual_saved_format_view':proof,
        'frozen_state_writer':dev.record(SOURCE/'bpe_streaming_training_state_v1.py'),
        'frozen_stream_writer':dev.record(SOURCE/'bpe_streaming_checkpoint_v1.py'),
        'actual_all207_tensor_restore_or_DEV_run':False,'historical_diagnostic_promoted':False,
        'GPU_audio_frontend_CACHE_or_model_job_started':False,'production_approval_claimed':False}
    dev.write_new(output/'report.json',report)
    print(json.dumps({'passed':True,'checks':len(checks),'format_only':True}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(Path(p.parse_args().output))
