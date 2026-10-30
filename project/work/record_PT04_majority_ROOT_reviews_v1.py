from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/parakeet_turkish';RUN='adaptive_20261012_TR_uniform_full_PT_04'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rec(p):
    with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'path':str(p.resolve()),'sha256':h,'bytes':p.stat().st_size}
closed=read(OUT/'TSC_uniform_full_PT04_7621_actual_CLOSED_ROOT_frontier_v1.json')
loop=OUT/'runs'/(RUN+'_loop_01');status=loop/'loop_status.json';state=read(status)
summary=OUT/'results'/f'{RUN}_step_7621'/'summary.json';counts=read(summary)
limits={'en':69,'es':52,'it':30,'fr':74,'de':70,'pt':66}
gates={l:counts['validation_'+l]['asr']['word_errors']<=n for l,n in limits.items()}
exit_record={'role':'trainer_controller','session_id':88291,'exit_code':1,'chunk_id':'b23736','observed_at_utc':closed['created_at_utc']}
base={'schema':1,'passed':True,'source_run_id':RUN,'chosen_step':7621,'chosen_checkpoint_role':'latest',
    'source_controller_sha256':'dfbac1e473274728f287169a9b3db3edeb03f39046d694c615c026ae7e2ada4f',
    'reviewed_status_artifact':rec(status),'chosen_DEV_summary':rec(summary),'owned_actual_exit':exit_record,
    'allow_majority_continuation':True,'guards_retained':True}
rows=[];event=None
with (loop/'memory_observations.jsonl').open(encoding='utf-8') as f:
    for line in f:
        row=json.loads(line);m=row['memory'];rows.append(row);rows=rows[-3:]
        if m.get('moderate_commit_streak')==3:
            event=[{'at_utc':x['at_utc'],'fraction':x['memory']['system_memory']['commit_fraction'],
                'global_commit_bytes':x['memory']['system_memory']['commit_used_bytes'],
                'owned_worker_private_bytes':x['memory']['worker_private_memory']['private_bytes'],
                'fresh':x['memory']['sample_fresh'],'fresh_interval_seconds':x['memory']['fresh_interval_seconds']} for x in rows]
            break
assert len(event)==3 and all(x['fraction']>=.9 and x['fresh'] and x['fresh_interval_seconds']>=5 for x in event)
resource={**base,'kind':'ROOT_TSC_uniform_majority_resource_review_v1','stop_reason':state['cooperative_stop_reason'],
    'material_policy_change':{'TR_per_update':14,'replay_per_update':2,'physical_microbatch_composition':[[8,0],[6,2]]},
    'actual_next_runtime_fit_not_claimed':True,'actual_three_fresh_trigger_samples':event,
    'observed_cause':'Whole-system commit crossed90percent for3fresh samples. GPU worker private was8.5-8.8GB attrigger; the global-minus-worker component rose substantially from earlier paired observations. Specific external process is unestablished.',
    'same_recipe_unchanged_resource_retry_allowed':False,
    'startup_memory_repair_and_fresh_admission_required':True,
    'pagefile_change_approved_or_applied':False}
retention={**base,'kind':'ROOT_TSC_uniform_majority_retention_review_v1','retention_gates':gates,
    'renewed_Portuguese_grace':False,'parent_failure_counter_preserved':True,
    'chosen_frontier_is_qualified_BEST':False,'parent_retention_failure_counter':2,
    'source_TR_word_errors':339,'source_TR_reference_words':825,
    'source_DE_word_errors':71,'source_DE_allowed':70,'source_PT_word_errors':67,'source_PT_allowed':66,
    'review_reason':'Latestfrontier is explicitly admitted for a changed Turkish-majority recovery trial. DE/PT each exceedstrictgate by1word, otherfourlanguagespass, and only3.53percentofnewuniformpassiscovered. TR41.09remainsworsethanold39.03champion. Thefrontier preservesactual8968newclips/allstate; noqualifiedBESTorfixedpasscompletionisclaimed.',
    'selection_and_goal_gates_relaxed':False,'first_new_DEV_must_recover_or_trigger_preserved_stop_counter':True}
for suffix,value in [('resource',resource),('retention',retention)]:
    p=OUT/f'TSC_uniform_majority_PT04_ROOT_{suffix}_review_v1.json';assert not p.exists();p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'review':rec(p),'owned_actual_exit':exit_record}))
