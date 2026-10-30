"""Prepare source37 finite metadata source/handoff only; no candidate import or proof execution."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import psutil

F=Path(__file__).parent
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
EVENT=F.parent/'resident_RAM_recurrent_metadata_slowdown_fix_v1/one_postDEV_gc_refresh_v1'
OLD=EVENT/'launch_source36_pre_stop_typed_metadata_external_v1.py'
NEW=F/'launch_source37_ninth_typed_metadata_external_v1.py'
FIXTURE=F/'verify_source37_ninth_entire_selector_and_inherited_event_metadata_v1.py'
LEDGER=F/'source37_ninth_typed_external_full_RAW_inverse_v1.json'
HANDOFF=F/'source37_ninth_entire_selector_frozen_CPU_handoff_v1.json'


def require(v,m):
    if not v:raise ValueError(m)


def pin(p):
    p=Path(p).resolve();require(p.stat().st_size<=8388608,'Compact source/metadata only')
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n')as h:
        json.dump(v,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())


def main():
    me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    print(json.dumps({'writer_identity':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},'writer_launcher':{'pid':me.parent().pid,'create_time':me.parent().create_time(),'ppid':me.parent().ppid()}}),flush=True)
    sample=sample_memory(me.pid,'source37_typed_source_handoff_preparation')
    write(F/'source37_typed_preparation_initial_numeric_sample_v2.json',sample)
    require(sample['paired_sampling_within_one_second']and sample['worker_private_memory']['private_bytes']<536870912
        and sample['system_memory']['commit_available_bytes']>=5055306368
        and sample['system_memory']['physical_available_bytes']>=2305306368
        and shutil.disk_usage('E:/').free>=100000000000,'Original full225/prefetch/prospective metadata guards')
    require(all(not p.exists()for p in (NEW,LEDGER,HANDOFF)),'Fresh immutable external/ledger/handoff targets')
    raw=OLD.read_bytes();text=raw.decode('utf-8');records=[]
    def replace(old,new,count=1):
        nonlocal text
        actualold=old;actualnew=new
        if text.count(actualold)!=count and '\n'in old:
            actualold=old.replace('\n','\r\n');actualnew=new.replace('\n','\r\n')
        require(text.count(actualold)==count,'Exact old external literal '+repr(old[:100]))
        records.append({'old_literal':actualold,'new_literal':actualnew,'expected_count':count})
        text=text.replace(actualold,actualnew)
    replace('One source36 entire-selector metadata proof','One source37 ninth/inherited-consumed entire-selector metadata proof')
    oldtoken='source36_pre_stop';newtoken='source37_ninth';replace(oldtoken,newtoken,text.count(oldtoken))
    replace('verify_source37_ninth_entire_selector_metadata_v1.py',FIXTURE.name)
    replace('ASR18_source37_ninth_entire_selector_metadata_CPU_v1','ASR18_source37_ninth_original_failure_entire_selector_metadata_CPU_v1')
    replace('protected_runtime_pids={3652,31268,30732,26892,30128}',
            'protected_runtime_pids={21912,37108,29600,39856,35440,26892,30128}')
    replace("    require(not RESULT.exists()and not OUTPUT.exists(),'ONE fresh actual observation, preserve every prior outcome')",
            "    require(not RESULT.exists()and not OUTPUT.exists()and not(F/'source37_ninth_external_initial_numeric_before_predicate_v1.json').exists(),'ONE fresh actual observation, preserve every prior outcome')")
    replace('        total=0\n        for proc in family:', '        total=0;family_samples=[]\n        for proc in family:')
    replace("                total+=sample['worker_private_memory']['private_bytes'];last=sample",
            "                total+=sample['worker_private_memory']['private_bytes'];last=sample;family_samples.append({'identity':identity(proc),'sample':sample})")
    replace('        prospective=LIMIT if admission else 0\n        require(total<=LIMIT',
            "        prospective=LIMIT if admission else 0\n        if admission:\n            initial_path=F/'source37_ninth_external_initial_numeric_before_predicate_v1.json'\n            with initial_path.open('x',encoding='utf-8',newline='\\n')as h:\n                json.dump({'family_samples':family_samples,'sampled_family_private_bytes':total,'E_free_bytes':shutil.disk_usage('E:/').free,\n                    'prospective_bytes':prospective,'commit_threshold':4518435456+prospective,\n                    'physical_threshold':1500000000+PREFETCH+prospective,'admission_predicate_evaluated':False},h,indent=2)\n                h.write('\\n');h.flush();os.fsync(h.fileno())\n        require(total<=LIMIT")
    replace("        initial=observe(admission=True)\n        with stdout.open('xb')as out,stderr.open('xb')as err:",
            "        initial=observe(admission=True)\n        for who in review['protected_runtime_identities']:\n            require(same_alive(who)is who['expected_alive'],'Exact protected training-dead/reader-alive identities, no rebind')\n        with stdout.open('xb')as out,stderr.open('xb')as err:")
    replace("        result['external_guard_error']=guard_error",
            "        initial_path=F/'source37_ninth_external_initial_numeric_before_predicate_v1.json'\n        result['initial_numeric_before_predicate']=pin(initial_path)if initial_path.exists()else None\n        result['external_guard_error']=guard_error")
    inverse=text
    for r in reversed(records):
        require(inverse.count(r['new_literal'])==r['expected_count'],'Unambiguous entire RAW external inverse')
        inverse=inverse.replace(r['new_literal'],r['old_literal'])
    require(inverse.encode('utf-8')==raw,'Full prior external bytes reconstructed exactly')
    ast.parse(text);ast.parse(FIXTURE.read_text(encoding='utf-8'))
    with NEW.open('xb')as h:h.write(text.encode('utf-8'));h.flush();os.fsync(h.fileno())
    write(LEDGER,{'kind':'source37_ninth_external_full_RAW_forward_inverse_v1','old_source':pin(OLD),'new_source':pin(NEW),
        'replacement_records':records,'whole_RAW_forward_inverse_exact':True,'newline_normalization':'NONE',
        'unchanged_fixed_limits':{'internal_seconds':100,'external_seconds':120,'sampled_full_family':536870912,
            'prospective':536870912,'admission_commit':5055306368,'admission_physical':2305306368,
            'runtime_commit':2268435456,'runtime_physical':1768435456,'E_free':100000000000},
        'new_initial_numeric_persisted_fsync_before_predicate_and_candidate_import':True,
        'exact_owned_cleanup_ancestry_parent_family_guard_algorithm_preserved':True})
    source_handoff=read(F/'source37_ninth_inactive_source_handoff_v1.json')
    pins=source_handoff['all_input_pins'][:]
    extras=[F/'source37_ninth_inactive_source_handoff_v1.json',FIXTURE,NEW,OLD,LEDGER,
        Path('E:/gpu_training_scratchpad/storage_config.json'),Path('E:/gpu_training_scratchpad/use_storage.ps1'),
        Path('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe'),
        S/'resource_watchdog_v2.py',S/'bpe_broad_audio_v1.py',S/'bpe_broad_production_state_v1.py',S/'atomic_io_retry_v1.py',
        Path(read(source_handoff['candidate_configuration']['path'])['resident_RAM_packet_credit_policy']['helper_source']['path']),
        Path(read(source_handoff['candidate_configuration']['path'])['resident_RAM_packet_credit_policy']['human_approval']['path'])]
    seen={p['path']for p in pins}
    for p in extras:
        v=pin(p)
        if v['path']not in seen:pins.append(v);seen.add(v['path'])
    # Known accepted closure metadata pin, copied from ROOT. Do not open it in preparation.
    if source_handoff['ninth_original_resume']['path']not in seen:pins.append(source_handoff['ninth_original_resume'])
    protected=[{'pid':21912,'create_time':1792780981.1092248,'expected_alive':False},
        {'pid':37108,'create_time':1792780981.1024935,'expected_alive':False},
        {'pid':29600,'create_time':1792780979.698954,'expected_alive':False},
        {'pid':39856,'create_time':1792790240.3032615,'expected_alive':False},
        {'pid':35440,'create_time':1792790240.2869232,'expected_alive':False},
        {'pid':26892,'create_time':1792652904.3359847,'expected_alive':True},
        {'pid':30128,'create_time':1792652904.283631,'expected_alive':True}]
    handoff={'kind':'source37_ninth_entire_selector_INACTIVE_finite_CPU_proposal_v1',
        'proposed_scope':'ASR18_source37_ninth_original_failure_entire_selector_metadata_CPU_v1',
        'fixture_source':pin(FIXTURE),'launcher_source':pin(NEW),'source_configuration':source_handoff['candidate_configuration'],
        'driver_source':source_handoff['driver_source'],'wrapper_source':source_handoff['wrapper_source'],
        'ninth_original_transition_validator':source_handoff['transition_validator'],
        'ninth_original_resume':source_handoff['ninth_original_resume'],'initial_checkpoint':source_handoff['initial_checkpoint'],
        'future_runtime7':source_handoff['future_runtime7'],'branch':str(Path('E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')),
        'output_path':str(F/'source37_ninth_typed_selector_actual_01'),
        'result_path':str(F/'source37_ninth_typed_metadata_external_result_v1.json'),
        'initial_numeric_before_predicate_path':str(F/'source37_ninth_external_initial_numeric_before_predicate_v1.json'),
        'protected_runtime_identities':protected,'input_pins':pins,
        'external_prefix':[str(Path('E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe')),'-Xutf8','-B',str(NEW)],
        'external_arguments':['--ROOT-review','PENDING_EXACT_ROOT_ONE_CPU_SCOPE_NO_EXECUTION_ADMITTED'],
        'full_RAW_external':pin(LEDGER),'full_RAW_driver_wrapper':source_handoff['full_RAW_AST_ledger'],
        'limits':{'internal_seconds':100,'external_seconds':120,'sampled_whole_family_private_bytes':536870912,
            'exact_external_launcher_parent_in_family':True,'full_additional_prospective_bytes':536870912,
            'admission_commit_bytes':5055306368,'admission_physical_bytes':2305306368,
            'runtime_commit_bytes':2268435456,'runtime_physical_bytes':1768435456,'E_free_bytes':100000000000,
            'CUDA_VISIBLE_DEVICES':'-1','all4math_thread_caps':1,'priority':'BelowNormal','retry_allowed':False},
        'coverage':['Real ninth policy/actual failure ROOT validators; exact f866 fivefield inverse and old proper36 runtime adapter',
            'Actual complete new entry/shared interval/preserved tail selectors and three native while ASTs independently ledgered exact',
            'Entire real journal_factory pre-restore dictionary and entire real postrestore dictionary, original25 STAT only',
            'Actual postcopy STAT predicate without copy, stale/None/wrong selectors and mixed historical lane negatives',
            'Complete actual runtime7 expression including BOTH serialized new configuration pins',
            'Entire actual wrapper entry/require/forwarding with explicit structural review METHOD_STUB',
            'Actual original helper2 constructor + actual driver inherited setup AST + actual helper metadata receipt + actual entire periodic-if skip',
            'No new event/capture/GC/sync/ineligible; completed/attempted false/operations empty; duplicate/attempted/completed/stale/exception negatives'],
        'scientific_authorities': 'No actual GPU/854 restore/state/draw/journal/copy/GC/capture/sync/audio/model; PrefixView/JournalBinding/structural review and forbidden event owners are explicit METHOD_STUBs.',
        'old_finite_or_audio_transport_proofs_reused_not_rerun_or_raw_report_read':True,
        'closed_163_tail_and_180k_event_provenance_ROOT_receipts_reused_no_raw_read':True,
        'known_actual_closure_pin_not_opened_during_source_preparation':True,
        'source37_production_or_recovery_execution_admitted':False,'actual_CPU_execution_admitted':False,
        'prep_identity':{'pid':me.pid,'create_time':me.create_time(),'ppid':me.ppid()},
        'prep_launcher_identity':{'pid':me.parent().pid,'create_time':me.parent().create_time(),'ppid':me.parent().ppid()}}
    write(HANDOFF,handoff)
    print(json.dumps({'fixture':pin(FIXTURE),'external':pin(NEW),'RAW_external':pin(LEDGER),'handoff':pin(HANDOFF),
        'input_count':len(pins),'prep_identity':handoff['prep_identity'],'prep_launcher_identity':handoff['prep_launcher_identity']}),flush=True)


if __name__=='__main__':main()
