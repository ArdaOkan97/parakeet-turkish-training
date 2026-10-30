"""Independent recovery-auditor regression fixtures; CPU only, new scratch files."""
from copy import deepcopy
from datetime import datetime, timezone
from contextlib import redirect_stderr
import io as io_module
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent.parent
OUTPUT=ROOT/'outputs/parakeet_turkish'
sys.path.insert(0,str(OUTPUT))
import verify_recovery_results_v1 as audit
from verify_final_results import AuditError


def main():
    directory=Path(tempfile.mkdtemp(prefix='recovery_auditor_cpu_',dir=ROOT/'work'))
    protected=['verify_adaptive_results.py','verify_final_results.py','benchmark.py','confidence.py',
               'metrics.py','runtime.py','train_lora_batched.py','batched_lora.py','efficient_tdt_v2.py',
               'benchmark_strong_tr_test.py','benchmark_strong_tr_test_v2.py','run_experiment_loop.py',
               'run_experiment_loop_recovery.py','atomic_io_retry_v1.py','launch_with_io_retry_v1.py',
               'sleep_recovery_quality_verification.json','large_batch_results_verification.json',
               'large_batch_speed_report.json','large_batch_matrix.json']
    before={name:audit.file_sha(OUTPUT/name) for name in protected}
    records=[];positive=0
    def passed(label,fn):
        nonlocal positive
        check=audit.Checks();value=fn(check);positive+=len(check.items)
        records.append({'name':label,'passed':True,'positive_checks':len(check.items)})
        return value
    def rejected(label,fn):
        try:fn(audit.Checks())
        except (AuditError,ValueError,KeyError,TypeError,IndexError) as error:
            records.append({'name':label,'passed':True,'rejected':True,'reason':str(error)})
        else:raise AssertionError('Corruption was accepted: '+label)
    def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

    smoke=audit.json_load(OUTPUT/'strong_longform_train_gpu_smoke_v2.json')
    fixture=audit.json_load(OUTPUT/'strong_test_v2_cpu_verification.json')['native_control_flow_fixture']
    passed('actual already completed49.08s TRAIN GPU output, CPU replay',lambda c:audit.verify_longform(smoke['longform'],smoke['generated_token_ids'],smoke['input_mel_frames'],c))
    passed('deterministic native CPU outer-loop fixture3684frames',lambda c:audit.verify_longform(fixture['longform'],fixture['aggregate_tokens'],3684,c))
    corruptions={
        'dropped final native window':lambda x:x['windows'].pop(),
        'skipped seek frame':lambda x:x['windows'][1].update(seek_start_frame=x['windows'][1]['seek_start_frame']+1),
        'truncated original mel length':lambda x:x['windows'][0].update(total_input_frames=3000),
        'incorrect tail valid length':lambda x:x['windows'][-1].update(input_frame_count=1),
        'zero native seek progress':lambda x:x['windows'][0].update(seek_advance_frames=0),
        'unbounded native seek progress':lambda x:x['windows'][0].update(seek_advance_frames=3001),
        'wrong decoder language prompt':lambda x:x['windows'][0].update(decoder_input_token_ids=[50258,50259,50360]),
        'missing raw window EOS':lambda x:x['windows'][0]['raw_generated_token_ids'].pop(),
        'false terminal EOS claim':lambda x:x['windows'][0].update(terminal_eos_emitted=False),
        'false complete tail claim':lambda x:x.update(full_input_consumed=False),
        'unobserved segment raw generation':lambda x:x['segments'][0]['raw_generated_token_ids'].__setitem__(3,1),
        'changed native segment slice':lambda x:x['segments'][0]['idxs'].__setitem__(0,4),
        'changed native segment content':lambda x:x['segments'][0]['tokens'].__setitem__(0,1),
        'negative segment time':lambda x:x['segments'][0].update(start=-.1),
        'segment outside observed window':lambda x:x['segments'][0].update(end=999.),
        'nonfinite segment time':lambda x:x['segments'][0].update(end=float('nan')),
    }
    for label,mutate in corruptions.items():
        evidence=deepcopy(smoke['longform']);mutate(evidence)
        rejected(label,lambda c,e=evidence:audit.verify_longform(e,smoke['generated_token_ids'],smoke['input_mel_frames'],c))
    rejected('changed aggregate actual generated IDs',lambda c:audit.verify_longform(smoke['longform'],smoke['generated_token_ids']+[1],smoke['input_mel_frames'],c))

    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'work/models/whisper-large-v3'),local_files_only=True)
    source={'audio_filepath':smoke['concatenated_wav_path'],'wav_sha256':smoke['concatenated_wav_sha256'],
            'duration':smoke['actual_duration_seconds']}
    row={key:deepcopy(smoke[key]) for key in ['generated_token_ids','hypothesis','input_samples','input_mel_frames',
         'input_attention_mask_frames','inference_mode','generation_protocol_sha256','longform']}
    row.update(eos_emitted=True,inference_seconds=smoke['elapsed_seconds'])
    passed('actual TRAIN raw-token decode and full-waveform route',lambda c:audit.verify_whisper_prediction(row,source,tokenizer,c))
    for key,value in [('input_samples',480000),('input_mel_frames',3000),('input_attention_mask_frames',3000),
                      ('inference_mode','short_form'),('generation_protocol_sha256','0'*64),('hypothesis','fabricated')]:
        bad=deepcopy(row);bad[key]=value
        rejected('long prediction rejects wrong '+key,lambda c,b=bad:audit.verify_whisper_prediction(b,source,tokenizer,c))
    devrow=audit.rows_load(OUTPUT/'results/strong_whisper_large_v3_greedy_dev_verified/validation_tr.jsonl')[0]
    devsource=next(s for s in audit.rows_load(ROOT/'work/data/validation_tr.jsonl') if s['source_id']==devrow['source_id'])
    import soundfile as sf
    samples=sf.info(devsource['audio_filepath']).frames
    protocol=audit.json_load(OUTPUT/'results/strong_whisper_large_v3_greedy_dev_verified/summary.json')['protocol']
    short={**devrow,'input_samples':samples,'input_mel_frames':3000,'input_attention_mask_frames':(samples+159)//160,
           'inference_mode':'short_form','generation_protocol_sha256':hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    passed('actual existing DEV raw-token short-form unchanged',lambda c:audit.verify_whisper_prediction(short,devsource,tokenizer,c))
    for mutate,label in [(lambda r:r['generated_token_ids'].__setitem__(3,50365),'changed short-form prompt'),
                         (lambda r:r['generated_token_ids'].pop(),'missing short-form EOS'),
                         (lambda r:r.update(longform={}), 'fabricated short-form segments')]:
        bad=deepcopy(short);mutate(bad)
        rejected(label,lambda c,b=bad:audit.verify_whisper_prediction(b,devsource,tokenizer,c))

    original=audit.json_load(OUTPUT/'runs/adaptive_20261010_01/loop_status.json')
    state={'started_at_utc':'2026-10-10T20:35:00+00:00','deadline_at_utc':original['deadline_at_utc']}
    passed('original absolute12h deadline retained after sleep',lambda c:audit.audit_deadline(state,original,c))
    for field,value,label in [('deadline_at_utc','2026-10-11T08:35:00+00:00','renewed12h recovery budget'),
                             ('started_at_utc',original['deadline_at_utc'],'recovery starts after expired deadline'),
                             ('started_at_utc','2026-10-10T00:00:00+00:00','recovery predates original loop'),
                             ('started_at_utc','2026-10-10T20:35:00','naive recovery timestamp')]:
        bad=deepcopy(state);bad[field]=value
        rejected(label,lambda c,b=bad:audit.audit_deadline(b,original,c))

    prior=next(t for t in original['trials'] if t['run_id'].endswith('screen_B'))
    experiment={'name':'screen_B2','parameters':{'head_lr':5e-5,'lora_lr':3e-4,'new_token_lr_multiplier':6},
                'initialization_weights':prior['initialization_weights'],'epochs_planned':.25,
                'status':'complete','exit_code':0,'epochs_completed':.25,
                'equal_requested_exposure_completed':True,'selection_provenance':{'synthetic_fixture':True},
                'selection':{'completed_steps':3204}}
    batch={**audit.json_load(OUTPUT/'runs'/prior['run_id']/'run_config.json'),'batch_size':2,'accumulation':2}
    passed('complete controlled equal-quarter B2 eligibility',lambda c:audit.audit_experimental_exposure(experiment,batch,prior,c))
    for field,value in [('epochs_completed',.249),('equal_requested_exposure_completed',False),('selection_rejected','incomplete'),
                        ('epochs_planned',.5),('name','extend_1'),('status','failed'),('exit_code',1),
                        ('initialization_weights',str(directory/'different.pt'))]:
        bad=deepcopy(experiment);bad[field]=value
        rejected('B2 rejects altered '+field,lambda c,b=bad:audit.audit_experimental_exposure(b,batch,prior,c))
    rejected('B2 effective exposure/physicalshape changed',lambda c:audit.audit_experimental_exposure(experiment,{'batch_size':4,'accumulation':1},prior,c))
    for key,value in [('seed',1),('steps',3203),('eval_epoch_fraction',.5),('warmup_steps',50),('replay_fraction',.25)]:
        bad=deepcopy(batch);bad[key]=value
        rejected('B2 controlled screen rejects changed '+key,lambda c,b=bad:audit.audit_experimental_exposure(experiment,b,prior,c))
    bad=deepcopy(experiment);bad['selection']['completed_steps']=3203
    rejected('B2 cannot qualify with fewer actual quarter updates',lambda c:audit.audit_experimental_exposure(bad,batch,prior,c))
    partial=deepcopy(experiment);partial.pop('selection_provenance');partial.update(selection_rejected='incomplete',epochs_completed=.12,equal_requested_exposure_completed=False)
    passed('rejected partial B2 evidence is retained but ineligible',lambda c:audit.audit_experimental_exposure(partial,batch,prior,c))
    captured={}
    def capture(path,filtered,base,check):captured['trials']=filtered['trials'];return 'fixture'
    with patch.object(audit.legacy,'check_frozen_selection',capture):
        full={'trials':[{'run_id':'accepted'},{'run_id':'partial','selection_rejected':'incomplete'}]}
        assert audit.check_frozen_selection(None,full,None,audit.Checks())=='fixture'
        assert captured['trials']==[{'run_id':'accepted'}] and len(full['trials'])==2
    records.append({'name':'incomplete rejected B2 excluded without mutating state','passed':True})

    evidence={'kind':'atomic_io_retry_v1','status':'complete','training_semantics_changed':False,'retry_winerrors':[5,32,33],
              'retry_seconds':30,'unique_same_directory_temporary':True,'fsync_before_replace':True,
              'worker':str((OUTPUT/'train_lora_batched.py').resolve()),
              'worker_sha256':audit.file_sha(OUTPUT/'train_lora_batched.py'),
              'launcher_sha256':audit.file_sha(OUTPUT/'launch_with_io_retry_v1.py'),
              'helper_sha256':audit.file_sha(OUTPUT/'atomic_io_retry_v1.py'),
              'arguments':['--','--run-id','cpu_fixture','--batch-size','1'],
              'started_at_utc':'2026-10-10T20:35:00+00:00','finished_at_utc':'2026-10-10T20:35:01+00:00',
              'retry_events':[{'path':str(directory/'consumed-private.tmp'),'retries':2,'elapsed_seconds':.03}]}
    for key in ['worker_sha256','launcher_sha256','helper_sha256']:evidence[key+'_after']=evidence[key]
    io=directory/'synthetic_io_provenance.json';write(io,evidence)
    passed('synthetic exact actual-source adapter metadata, consumed temp allowed',lambda c:audit.audit_io(io,'train_lora_batched.py','cpu_fixture',c,expected_arguments=['--run-id','cpu_fixture','--batch-size','1']))
    for key,value in [('status','failed'),('training_semantics_changed',True),('retry_winerrors',[5,32,33,2]),
                      ('retry_seconds',61),('unique_same_directory_temporary',False),('fsync_before_replace',False),
                      ('worker_sha256_after','0'*64),('launcher_sha256','0'*64),('helper_sha256_after','0'*64),
                      ('arguments',['--run-id','other']),('finished_at_utc','2026-10-10T20:34:00+00:00')]:
        bad=deepcopy(evidence);bad[key]=value;write(io,bad)
        rejected('adapter rejects altered '+key,lambda c:audit.audit_io(io,'train_lora_batched.py','cpu_fixture',c))
    write(io,evidence)
    rejected('adapter does not permit changed argv',lambda c:audit.audit_io(io,'train_lora_batched.py','cpu_fixture',c,expected_arguments=['--run-id','cpu_fixture','--batch-size','2']))

    oldrun='adaptive_20261010_01_screen_B';old_dir=OUTPUT/'runs'/oldrun
    oldfinal=audit.json_load(old_dir/'selection.json');oldmarker=audit.json_load(old_dir/'finalization_provenance.json')
    quality=OUTPUT/'sleep_recovery_quality_verification.json'
    reused={'run_id':oldrun,'selected_step':oldfinal['selected_step'],'quality_verification_path':str(quality),
            'quality_verification_sha256':audit.file_sha(quality),'final_selection_sha256':audit.file_sha(old_dir/'selection.json'),
            'finalization_provenance_sha256':audit.file_sha(old_dir/'finalization_provenance.json'),
            'candidate_files_sha256':oldmarker['candidate_files_sha256'],'policy':'CPU fixture, actual preserved source binding'}
    write(directory/'reused_original_finalization.json',reused)
    reuse_state={'recovery':{'original_loop_id':'adaptive_20261010_01','original_status_snapshot':str(directory/'original_loop_status.json')},
                 'reused_original_finalization':reused}
    frozen={'run_id':oldrun,'selected_step':oldfinal['selected_step']}
    passed('actual carried B reuse27 preserved files unchanged',lambda c:audit.audit_carried_reuse(reuse_state,frozen,c))
    for key,value in [('quality_verification_sha256','0'*64),('final_selection_sha256','0'*64),
                      ('finalization_provenance_sha256','0'*64),('candidate_files_sha256',{})]:
        bad=deepcopy(reuse_state);bad['reused_original_finalization'][key]=value
        write(directory/'reused_original_finalization.json',bad['reused_original_finalization'])
        rejected('reuse rejects altered '+key,lambda c,b=bad:audit.audit_carried_reuse(b,frozen,c))
    write(directory/'reused_original_finalization.json',reused)
    bad=deepcopy(reuse_state);bad['finalizer_io_retry_provenance_path']='fabricated'
    rejected('reused B cannot claim newly executed finalizer',lambda c:audit.audit_carried_reuse(bad,frozen,c))
    rejected('fresh winner cannot claim carried B reuse',lambda c:audit.audit_carried_reuse(reuse_state,{'run_id':'other','selected_step':3204},c))

    # The carried-B whole-recovery branch uses actual original snapshots and
    # preserved evidence; the wrapper metadata above remains a separate fixture.
    original_dir=OUTPUT/'runs/adaptive_20261010_01'
    (directory/'original_loop_status.json').write_bytes((original_dir/'loop_status.json').read_bytes())
    (directory/'original_frozen_final_selection.json').write_bytes((original_dir/'frozen_final_selection.json').read_bytes())
    oldfreeze=audit.json_load(original_dir/'frozen_final_selection.json')
    config=audit.json_load(old_dir/'run_config.json');selected=audit.configuration(config)
    carried={**deepcopy(prior),'carried_from_original_loop':True,'selection_provenance':oldfreeze['provenance'],
             'selected_configuration':selected,'batch_numerical_equivalence_proven':False,
             'existing_training_configuration_verified':True}
    frozen={**oldfreeze,'selected_configuration':selected}
    complete_state={**state,'trials':[carried],'best_development_candidate':frozen,
        'reused_original_finalization':reused,'recovery':{
            'controller_path':str(OUTPUT/'run_experiment_loop_recovery.py'),
            'controller_sha256':audit.file_sha(OUTPUT/'run_experiment_loop_recovery.py'),
            'original_loop_id':'adaptive_20261010_01','original_status_snapshot':str(directory/'original_loop_status.json'),
            'original_status_snapshot_sha256':audit.file_sha(directory/'original_loop_status.json'),
            'original_frozen_selection_snapshot_sha256':audit.file_sha(directory/'original_frozen_final_selection.json'),
            'io_retry_launcher_sha256':audit.file_sha(OUTPUT/'launch_with_io_retry_v1.py'),
            'io_retry_helper_sha256':audit.file_sha(OUTPUT/'atomic_io_retry_v1.py'),
            'strong_test_worker':'benchmark_strong_tr_test_v2.py','audit_worker':'verify_recovery_results_v1.py',
            'previous_output_artifacts':{}}}
    passed('whole recovery carried-B branch with actual preserved sources',lambda c:audit.audit_recovery(complete_state,frozen,config,c))
    for mutate,label in [
        (lambda s:s.update(deadline_at_utc='2026-10-11T08:35:00+00:00'),'whole recovery rejects renewed deadline'),
        (lambda s:s['recovery'].update(controller_sha256='0'*64),'whole recovery rejects controller SHA drift'),
        (lambda s:s['recovery'].update(original_status_snapshot_sha256='0'*64),'whole recovery rejects prior snapshot drift'),
        (lambda s:s['trials'][0].update(parameters={}), 'whole recovery rejects carried B learning-parameter change'),
        (lambda s:s['trials'][0].update(selection_provenance={}), 'whole recovery rejects carried B DEV binding drift'),
        (lambda s:s['trials'][0].update(batch_numerical_equivalence_proven=True), 'whole recovery rejects false carried B BF16 equivalence'),
        (lambda s:s.pop('reused_original_finalization'), 'old B must reuse rather than execute a new finalizer'),
    ]:
        bad=deepcopy(complete_state);mutate(bad)
        rejected(label,lambda c,b=bad:audit.audit_recovery(b,frozen,config,c))

    numeric=passed('actual immutable30-case numeric classification preserved',audit.audit_batch_numerical_evidence)
    assert numeric['fp32_passed_cases']==8 and numeric['bf16_strict_failed_cases']==10
    numerical_path=OUTPUT/'large_batch_results_verification.json'
    real_json_load=audit.json_load;actual_numerical=real_json_load(numerical_path)
    for kind in ['waived BF16 failure','changed FP32 tolerance','drifted matrix hash','false integrity success']:
        changed=deepcopy(actual_numerical)
        if kind=='waived BF16 failure':
            next(row for row in changed['profiles'] if row['mode']=='timing')['numerical_comparison']['strict_gate']['passed']=True
        elif kind=='changed FP32 tolerance':
            next(row for row in changed['profiles'] if row['mode']=='fp32-parity')['numerical_comparison']['strict_gate']['limits']['gradient_relative_l2']=.1
        elif kind=='drifted matrix hash':changed['matrix_sha256_at_audit']='0'*64
        else:changed['passed']=False
        def injected(path):
            return changed if Path(path).resolve()==numerical_path.resolve() else real_json_load(path)
        with patch.object(audit,'json_load',injected):
            rejected('numerical classification rejects '+kind,audit.audit_batch_numerical_evidence)
    original_files={str(path) for path in original_dir.rglob('*') if path.is_file()}
    for target,label in [(original_dir,'old loop cannot receive failed or successful audit outputs'),
                         (old_dir,'winner directory cannot receive loop audit outputs'),
                         (directory,'external work directory cannot receive loop audit outputs')]:
        args=SimpleNamespace(frozen_selection=original_dir/'frozen_final_selection.json',
                             strong_test_comparison=directory/'not_needed.json',baseline_results=OUTPUT/'results/base_v3',
                             output_dir=target)
        with redirect_stderr(io_module.StringIO()):
            try:audit.main(args)
            except SystemExit as error:assert error.code==2
            else:raise AssertionError('Unsafe audit output destination accepted')
        records.append({'name':label,'passed':True,'rejected':True})
    assert original_files=={str(path) for path in original_dir.rglob('*') if path.is_file()}
    result={'schema':1,'passed':True,'gpu_used':False,'new_ASR_inference_performed':False,
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'auditor_sha256':audit.file_sha(Path(audit.__file__)),
            'guard_script_sha256':audit.file_sha(Path(__file__)),'cases':len(records),'positive_checks':positive,
            'corruptions_rejected':sum(r.get('rejected',False) for r in records),'checks':records,
            'protected_source_and_evidence_sha256':before,'all_protected_bytes_unchanged':True,
            'actual_train_smoke_sha256':audit.file_sha(OUTPUT/'strong_longform_train_gpu_smoke_v2.json'),
            'actual_native_cpu_fixture_sha256':audit.file_sha(OUTPUT/'strong_test_v2_cpu_verification.json'),
            'scope':'Actual previously saved TRAIN49.08s long-form output and existing short DEV output are replayed on CPU. Corruptions and I/O metadata are synthetic fixtures. Carried B hashes and30-case numerical gate evidence are actual. Full final900-recording recovery audit requires a completed frozen recovery winner and strong TESTv2.'}
    assert before=={name:audit.file_sha(OUTPUT/name) for name in protected}
    write(OUTPUT/'recovery_auditor_cpu_verification.json',result)
    print(json.dumps({key:result[key] for key in ['passed','cases','positive_checks','corruptions_rejected','auditor_sha256','all_protected_bytes_unchanged']},indent=2))


if __name__=='__main__':main()
