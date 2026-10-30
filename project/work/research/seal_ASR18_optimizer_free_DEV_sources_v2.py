"""Seal source-only optimizer-free DEV work and genuine observed control exits."""
import ast
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT/'outputs/parakeet_turkish'
RESEARCH = PROJECT/'work/research'
ROOT = Path(r'E:\gpu_training_scratchpad')
AREA = ROOT/'analysis/ASR18_memory_diagnosis_v1'
sys.path.insert(0,str(SOURCE))


def record(path):
    path = Path(path).resolve()
    assert path.is_relative_to(PROJECT.resolve()) or path.is_relative_to(ROOT.resolve())
    before=path.stat(); assert before.st_size < 16*1024**2, 'Small metadata/source only, never payload'
    raw=path.read_bytes(); assert (before.st_size,before.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
    return record(path)


def main():
    from resource_watchdog_v2 import sample_memory
    samples=[]
    for index in range(3):
        value=sample_memory(os.getpid(),'optimizer_free_DEV_source_seal')
        system=value['system_memory'];worker=value['worker_private_memory']
        assert system['available'] and worker['available'] and value['paired_sampling_within_one_second']
        assert worker['private_bytes']+64_000_000<2_000_000_000
        assert (system['commit_used_bytes']+64_000_000)/system['commit_limit_bytes']<.90
        assert system['commit_limit_bytes']-system['commit_used_bytes']-64_000_000>2_000_000_000
        assert shutil.disk_usage(ROOT).free>100_000_000_000
        samples.append(value);time.sleep(.05)
    entry_names=('asr18_optimizer_free_DEV_structure_v2.py','asr18_full_run_serial_DEV_v6.py','asr18_full_run_DEV_runtime_v3.py')
    report=read(AREA/'DEV_lifecycle_saved_text_contract_actual_01/report.json')
    guard=read(AREA/'DEV_lifecycle_saved_text_contract_guard_01/closure.json')
    assert report['passed'] is True and report['check_count']==18 and guard['actual_exit_code']==0
    assert report['producer']['pid']==guard['pid'] and guard['monitor_clean'] and guard['monitor_joined']
    assert guard['observed']['samples']>=3 and guard['observed']['monitor_error'] is None
    for name,pin in report['sources'].items():assert record(SOURCE/name)==pin
    actual=write_new(AREA/'DEV_lifecycle_saved_text_contract_actual_01/actual_owned_CLOSED0.json',{
        'kind':'ASR18_source_bound_owned_metadata_control_actual_TOOL0_v1','owner':'/root/memory_diagnosis',
        'actual_exit_code':0,'actual_tool_chunk_id':'fb4122','native_PID':guard['pid'],
        'source':record(RESEARCH/'validate_ASR18_DEV_lifecycle_saved_text_contract_v1.py'),
        'report':record(AREA/'DEV_lifecycle_saved_text_contract_actual_01/report.json'),
        'native_guard_closure':record(AREA/'DEV_lifecycle_saved_text_contract_guard_01/closure.json'),
        'operator_observed_actual_tool_exit':True,'ML_audio_payload_GPU_executed':False})
    old=record(AREA/'optimizer_free_DEV_metadata_actual_CLOSED1_preserved_v1.json')
    prior_bundle=record(AREA/'optimizer_free_serial350DEV_source_review_manifest_v1.json')
    prior_controls=record(AREA/'optimizer_free_DEV_metadata_actual_02/actual_owned_CLOSED0.json')
    # Static owned-import closure. Dynamic resource/native-owner sources remain
    # required in the future ROOT-reviewed configuration; none is invented here.
    queue=deque(SOURCE/name for name in entry_names)
    queue.extend((RESEARCH/'validate_ASR18_DEV_lifecycle_saved_text_contract_v1.py',
                  RESEARCH/'ASR18_bounded_CPU_phase_v3.py',Path(__file__)))
    snapshots={}
    while queue:
        path=queue.popleft().resolve()
        if str(path) in snapshots:continue
        tree=ast.parse(path.read_text(encoding='utf-8-sig'));compile(tree,str(path),'exec')
        snapshots[str(path)]=record(path)
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[item.name for item in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:names=[node.module]
            for name in names:
                if '.' in name:continue
                for directory in (SOURCE,RESEARCH):
                    candidate=directory/(name+'.py')
                    if candidate.is_file():queue.append(candidate);break
    from asr18_full_run_DEV_runtime_v3 import resource_proposal
    base=Path((PROJECT/'work/base_model_path.txt').read_text().strip()).resolve()
    assets={name:record(base/name) for name in ('config.json','generation_config.json','processor_config.json','tokenizer_config.json','tokenizer.json')}
    membership=record(AREA/'ROOT_fixed_serial350DEV_manifest_membership_review_v1.json')
    original_DEV=record(AREA/'feature_window_native20_DEV_01/actual_DEV_summary_v1.json')
    manifest_bindings={str(Path(pin['path'])):pin for pin in snapshots.values()}
    for pin in assets.values():manifest_bindings[pin['path']]=pin
    for path in (AREA/'ROOT_fixed_serial350DEV_manifest_membership_review_v1.json',
                 AREA/'feature_window_native20_DEV_01/actual_DEV_summary_v1.json',
                 ROOT/'analysis/FastMSS_integration_v1/DEV_selective_weight_RNG_component_source_manifest_v2.json'):
        pin=record(path);manifest_bindings[pin['path']]=pin
    canonical_paths={language:PROJECT/'work/data'/('portuguese_retention_v1/' if language=='pt' else '')/('validation_'+language+'.jsonl')
                     for language in ('tr','en','es','it','fr','de','pt')}
    fixed_DEV={language:record(path) for language,path in canonical_paths.items()}
    for pin in fixed_DEV.values():manifest_bindings[pin['path']]=pin
    for pin in (actual,old,prior_bundle,prior_controls):manifest_bindings[pin['path']]=pin
    interface={'kind':'ASR18_optimizer_free_serial350DEV_source_interface_v2','owner':'/root/memory_diagnosis',
        'production_enabled':False,'execution_admitted':False,'entrypoints':{name:record(SOURCE/name) for name in entry_names},
        'structure_API':'prepare_structure(model,processor,torch,guard,admit) -> original8711 tokenizer + fifteen head/decoder/projector +192LoRA; all temporary207 weights must be overwritten before forward',
        'runtime_API':'run(config_path,folder,*,parent_source,check_bindings,storage,stage,admit_stage); no launcher/CLI or default admission',
        'DEV_API':'evaluate_boundary_checkpoint(...); SerialDEVCallbacks(...,native_boundary_sources,...) uses source-bound future execute_owned and native_boundary callbacks',
        'configuration_kind':'ASR18_optimizer_free_serial350DEV_worker_configuration_v1',
        'ROOT_execution_review_kind':'ROOT_optimizer_free_serial350DEV_execution_configuration_review_v1',
        'ROOT_resource_review_kind':'ROOT_optimizer_free_serial350DEV_resource_review_v1',
        'ROOT_scorer_policy_review_kind':'ROOT_selective_optimizer_free_serial350DEV_policy_and_sources_review_v2',
        'required_parent_callbacks':'All four callback sourcefiles must exactly equal reviewed parent_source; actual parent must implement launch, fresh3paired whole-worker+parent/shim admission, continuous guards, sole lease, actual PID handshake, strict actual0 sealing',
        'admission_handshake':'Pre-worker immutable request; worker emits real producer; parent separately publishes owned_worker_admission.json including actual producer, request, ROOT execution/resource pins and fresh whole forecast. No future-PID binding or fake tool chunk',
        'native_boundary_kind':'ASR18_full_run_native_exit_and_all_checkpoint_recovery_boundary_for_serial_DEV_v1',
        'required_native_boundary_sources':['boundary_verifier','native_worker','native_parent','recovery_worker','recovery_parent'],
        'native_boundary_contract':'All actual closed non-diagnostic native process/window records and ALL captured checkpoint entries (<=6 in bounded tail chunk), exact finalpointer/identity/globalclock/sampler/prefix, distinct recovery actualprocess0 with every ordered207/threeAdam/all4RNG/samplers/prefix/nextdraw recovery. No live idle/persistence-only/finite20/21/40 promotion',
        'source_bound_recovery_report_kind':'ASR18_full_run_actual_all_captured_checkpoints_recovered_v1',
        'saved_weight_restore':'Frozen WeightOnlyCheckpoint selects207FP32weights +Torch/CUDA RNG only, all rawpayload integrity hash; scalar Adam clocks/sampler/Python/NumPy metadata only. NoAdam tensorviews/objects/sampler/replay construction;207 GPUhashes exact before forward, all4 RNG exact; readonly mapping closed before evaluation',
        'original_tokenizer_and_generation':'Original8193IDs preserved in frozen8711candidate; blank8192; originalfive duration suppression8193..8197 moves directly to8711..8715; no new lexical suppression; no donor checkpoint needed because all207 overwritten',
        'original_fixed_DEV':fixed_DEV,'membership_review':membership,'base_assets':assets,
        'official_reference_words':dict(tr=825,en=1045,es=1289,it=1158,fr=1072,de=1004,pt=1095),
        'retention_integer_caps':dict(en=69,es=52,it=30,fr=74,de=70,pt=66),
        'original_evaluator':'Unchanged trainer.evaluate/gate, actual saved350 text independently rescored by frozen lifecycle. TEST unused; all207 hashes/all4RNG unchanged afterDEV; sourcecheckpoint/sampler/prefix rechecked',
        'resource_proposal':resource_proposal(),'candidate_resource_accepted':False,
        'resource_scope':'5.45GB candidate DEV LEAF ceiling retains3GB pre-restore+350MBrestore+2.1GBDEV; parent+launcher and whole-job global admission remain additional. No fullfit/throughput inference from metadata controls',
        'actual_new_controls':actual,'preserved_initial_failure':old,'historical_closed350_derivation_only':original_DEV,
        'pending':['ROOT exact component/source review','Concrete reviewed native-owner DEV launcher/continuous full-family guard/actual0 executor',
            'Fullrun-compatible native+allcheckpoint recovery boundary publisher and sources (current finite diagnostic workers explicitly rejected)',
            'ROOT exact scorer policy/runtime/resource/config and fresh admission','Separately reviewed bounded diagnostic use of SAME optimizer-free/selective restore/evaluator primitives, unqualified; fullrun hook correctly rejects finite20/21/40 and cannot itself supply that pre-production proof',
            'Durable lifecycle DEV-ack/winner/resume integration'],
        'closed_controls_scope':'NEW18 focused historical saved-text official_DEV and real callback-consumer contract; synthetic process IO only, no new model/audio/DEV/native proof. Prior33 controls and initial metadata1 preserved.',
        'canonical_lifecycle_report_kind':'ASR18_serial_complete350_boundary_DEV_v1',
        'required_optimizer_free_protocol':'ASR18_selected207_all4RNG_zeroAdam_serialDEV_protocol_v1',
        'previous_reportkind_blocker_preserved_in_bundle':prior_bundle,
        'prior33_controls':prior_controls,
        'frozen_caller9_CPU6_pilot_policy_unchanged':True,'existing_admission04_not_retried':True}
    interface_pin=write_new(AREA/'optimizer_free_serial350DEV_source_interface_v2.json',interface)
    manifest_bindings[interface_pin['path']]=interface_pin
    result={'kind':'ASR18_optimizer_free_serial350DEV_source_review_manifest_v2','owner':'/root/memory_diagnosis',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'entrypoints':interface['entrypoints'],
        'interface':interface_pin,'bindings':manifest_bindings,'owned_AST_source_count':len(snapshots),
        'binding_count':len(manifest_bindings),'actual_source_contract_controls':actual,'preserved_initial_failure':old,
        'all_bindings_verified':True,'all_owned_sources_AST_compiled':True,'actual_original_DEV_executed':False,
        'ROOT_reviewed_or_execution_approved':False,'production_enabled':False,
        'prospective_metadata_seal_increment_bytes':64_000_000,'fresh_three_samples':samples}
    for pin in manifest_bindings.values():assert record(pin['path'])==pin
    manifest=write_new(AREA/'optimizer_free_serial350DEV_source_review_manifest_v2.json',result)
    # Mutable dispatch moves to a NEW immutable readiness snapshot; pinned history
    # and the failed admission stay untouched and no actual GPU lease is changed.
    pointer_path=AREA/'current_launch_readiness.json';pointer=read(pointer_path)
    previous=pointer['current_readiness'];status=read(previous['path'])
    status.update(kind='ASR18_compact_memory_and_production_preparation_readiness_v58',
        updated_at_utc=datetime.now(timezone.utc).isoformat(),previous_history=previous,
        current_stage='GPU_idle_admission04_denied_DEV6_saved_text_callback_contract_fixed_memory_remedy_preparation',
        inactive_optimizer_free_DEV_source_manifest=manifest,inactive_optimizer_free_DEV_interface=interface_pin,
        actual_GPU_running=False,actual_owned_GPU_PID=None,current_CPU_metadata_child=None,
        production_enabled=False,production_ready=False,
        remaining='Admission04 remains denied/no unchanged retry. NEW optimizer-free DEV6/runtime3 canonical report/protocol +focused18 saved-text/callback controls ready for exact ROOT review; parent executor, fullrun native/recovery boundary, numeric/scorer/config review and actual ML350 remain pending. No audio/model/payload/GPU job or frozen caller change.')
    status_pin=write_new(AREA/'launch_readiness_v58.json',status)
    pointer.update(updated_at_utc=status['updated_at_utc'],current_stage=status['current_stage'],current_readiness=status_pin,
        actual_GPU_running=False,actual_owned_GPU_PID=None,current_CPU_metadata_child=None,
        inactive_optimizer_free_DEV_source_manifest=manifest)
    temporary=pointer_path.with_suffix('.updating.json');temporary.write_text(json.dumps(pointer,indent=2),encoding='utf-8');temporary.replace(pointer_path)
    for name in ('memory_diagnosis_status.json','production_preparation_status_v1.json'):
        (AREA/name).write_text(json.dumps(pointer,indent=2),encoding='utf-8')
    print(json.dumps({'manifest':manifest,'interface':interface_pin,'status':status_pin,
        'actual_controls':actual,'binding_count':len(manifest_bindings),'owned_sources':len(snapshots),
        'GPU_used':False,'production_enabled':False}))


if __name__=='__main__':main()
