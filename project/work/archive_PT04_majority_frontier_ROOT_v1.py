"""Archive one actually CLOSED full-state frontier, preserving all source bytes."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/parakeet_turkish'
RUN='adaptive_20261012_TR_uniform_full_PT_04'
DIRECTORY=ROOT/'work/closed-sources/uniform_PT04_7621_latest_v1'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rec(p):
    with p.open('rb') as f:d=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':d}
loop=OUT/'runs'/(RUN+'_loop_01');run=OUT/'runs'/RUN
state=read(loop/'loop_status.json');selection=read(run/'selection.json')
done=read(loop/(RUN+'_native_backend.completion.json'))
assert state['status']=='resource_review_required' and state['child_exit_code']==0
assert selection['completed_steps']==7621 and selection['selected_step'] is None
assert done['status']=='complete' and done['sources_unchanged'] and done['pid']==37732
coverage=selection['Turkish_domain_coverage']['uniform_pass']
assert coverage['unique_clips_seen']==coverage['draws']==8968 and coverage['repeat_visits']==0
assert not DIRECTORY.exists();DIRECTORY.mkdir(parents=True)
sources={'chosen_checkpoint':ROOT/'work'/f'{RUN}_latest.pt','selection':run/'selection.json',
    'controller_status':loop/'loop_status.json','native_ready':loop/(RUN+'_native_backend.json'),
    'native_completion':loop/(RUN+'_native_backend.completion.json'),'owned_identity':loop/(RUN+'_identity.json'),
    'memory':loop/(RUN+'_memory.json'),'cache':loop/(RUN+'_cache.json'),'io':loop/(RUN+'_io.json'),
    'plan':loop/'plan.json','run_config':run/'run_config.json','history':run/'training_history.json',
    'initial350':run/'uniform_full_initial_source_equivalence.json','indexed_TR_audit':run/'indexed_TR_audit.json',
    'initial_startup':run/'uniform_full_startup_from_6500.json',
    'dispatcher':OUT/'run_TSC_uniform_full_v1.py','worker':OUT/'train_lora_batched_tsc_uniform_full_v1.py',
    'DEV_summary':OUT/'results'/f'{RUN}_step_7621'/'summary.json',
    'DEV_metadata':OUT/'results'/f'{RUN}_step_7621'/'metadata.json'}
artifacts={}
for name,p in sources.items():
    before=rec(p);destination=DIRECTORY/(name+p.suffix);shutil.copyfile(p,destination);after=rec(p);copy=rec(destination)
    assert before==after and copy['bytes']==before['bytes'] and copy['sha256']==before['sha256']
    artifacts[name]={'original':before,'archived':copy}
dev=read(sources['DEV_summary']);limits={'en':69,'es':52,'it':30,'fr':74,'de':70,'pt':66}
counts={lang:dev['validation_'+lang]['asr'] for lang in ('tr','en','es','it','fr','de','pt')}
assert all(x['utterances']==50 for x in counts.values())
gates={lang:counts[lang]['word_errors']<=limit for lang,limit in limits.items()}
receipt={'schema':1,'kind':'ROOT_actual_closed_full_uniform_majority_frontier_v1','status':'actual_CLOSED',
    'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_run_id':RUN,
    'actual_owned_ROOT_session':88291,'actual_owned_ROOT_exit_code':1,'actual_owned_ROOT_exit_tool_chunk':'b23736',
    'actual_child_exit_code':0,'actual_native_pid':37732,'actual_source_step':7621,'chosen_step':7621,
    'chosen_checkpoint_role':'LATEST_recovery_frontier_not_qualified_BEST','selected_step':None,
    'resource_stop_reason':state['cooperative_stop_reason'],'preserved_native_source_unchanged':True,
    'artifacts':artifacts,'new_uniform_pass':coverage,
    'parent_retention_failures':selection['consecutive_retention_failures'],
    'actual_complete_DEV_counts':{l:{k:x[k] for k in ('word_errors','reference_words','utterances')} for l,x in counts.items()},
    'actual_six_language_gate_map':gates,'all_six_qualified':all(gates.values()),
    'history_sampler_RNG_Adam_same_chosen_file':True,'actual_tensor_deserialization_performed':False,
    'remaining_TR_clips':253798-8968,'proposed_14TR2RP_remaining_updates':17488,
    'proposed_global_end_step':25109,'no_new_uniform_pass':True,'no_renewed_PT_grace':True,
    'checkpoint_archived_byte_identical':True,'every_historical_metric_preserved':True,
    'no_GPU_launched':True,'Torch_imported':'torch' in sys.modules,'goal_complete':False}
dest=OUT/'TSC_uniform_full_PT04_7621_actual_CLOSED_ROOT_frontier_v1.json';assert not dest.exists()
dest.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'receipt':rec(dest),'archived_checkpoint':artifacts['chosen_checkpoint']['archived'],'actual_gates':gates}))
