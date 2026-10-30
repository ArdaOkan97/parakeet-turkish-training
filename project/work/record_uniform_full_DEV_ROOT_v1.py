"""Review complete observed DEV files and update the ordered plan, no runtime edits."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/parakeet_turkish'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rec(path):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':digest}
path=OUT/'experiment_plan.json';plan=read(path);item=next(x for x in plan['ordered_experiments'] if x['id']=='04')
run=item['run_id'];assert run=='adaptive_20261012_TR_uniform_full_PT_04'
queue=read(OUT/'experiment_queue_state.json');assert queue['current_run_id']==run and queue['status']=='running'
limits=plan['retention_max_fixed_DEV_word_errors'];points=[]
for step in (6750,7000):
    summary_path=OUT/'results'/f'{run}_step_{step}'/'summary.json';summary=read(summary_path)
    counts={lang:summary['validation_'+lang]['asr'] for lang in ('tr','en','es','it','fr','de','pt')}
    assert all(x['utterances']==50 for x in counts.values())
    gates={lang:x['word_errors']<=limits[lang] for lang,x in counts.items() if lang!='tr'}
    points.append({'step':step,'new_uniform_updates':step-6500,'TR_WER':counts['tr']['wer'],
        'actual_counts':{lang:{k:x[k] for k in ('word_errors','reference_words','utterances')} for lang,x in counts.items()},
        'retention_language_passed':gates,'all_six_passed':all(gates.values()),
        'original_five_passed':all(gates[x] for x in ('en','es','it','fr','de')),
        'strict_all_six_champion_eligible':all(gates.values()),'summary':rec(summary_path)})
metrics_path=OUT/'Trackio_V11_PT04_actual_metrics_verification_v1.json';metrics=read(metrics_path)
receipt={'kind':'ROOT_actual_PT04_complete_DEV250_500_review_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
    'run_id':run,'actual_complete_DEV_points':points,'queue_step_at_review':queue['published_step'],
    'sole_ROOT_owned_controller_session':88291,'actual_GPU_training_continues':True,
    'new_pass_not_complete':True,'all_six_champion_yet':False,'TEST_used':False,
    'independent_dashboard_and_rates':rec(metrics_path),
    'latter100updates_through6750':{'mixed_audio_min_per_train_min':84.25965,'TR_audio_min_per_train_min':21.13829},
    'rate_comparison_scope':'Full encoder LoRA/uniform pass; earlier116/42 came from frozen-encoder row-only and30percentFLEURS, not a matched speed gain.',
    'next_action':'Leave healthy soleGPU child running. Originalfive recovered atDEV500; PT remains within declared1000-new-update grace. EverynewBEST requires allsix. No recipe/source edits.'}
dest=OUT/'TSC_uniform_full_PT04_DEV250_500_ROOT_review_v1.json';assert not dest.exists();dest.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
item.update(status='running',actual_new_uniform_updates=queue['actual_new_uniform_updates'],actual_GPU_training_started=True,
    actual_DEV_points=points,actual_full_pass_completed=False,initial_all_six_champion_claimed=False,
    actual_initial_state_serialization_completed=True,actual_initial_state_roundtrip_claimed=False,
    actual_runtime_memory_observed=True,actual_complete_dataset_memory_fit_claimed=False,
    latest_original_five_passed=True,latest_Portuguese_passed=False,
    first_250_Portuguese_passed=True,all_six_champion_yet=False,
    early_measured_peak_owned_private_bytes=10698612736,early_measured_peak_vram_gb=5.903609344,
    actual_phase_throughput={'mixed_audio_min_per_train_min':82.01095,'TR_audio_min_per_train_min':20.72948},
    latest100updates_through6750={'mixed_audio_min_per_train_min':84.25965,'TR_audio_min_per_train_min':21.13829},
    actual_DEV_review=dest.name,
    pending=['Complete253798 new-pass unique clips with periodic fixed seven-language DEV',
             'Strict all-six qualifying champion and final quality target',
             'Actual exact checkpoint restore at first normal budget closure'])
plan['status_authority']='Actual controller/native closure and complete fixed DEV are authoritative. FullPT04 is advancing; DEV250 and500 actually complete, no all-six winner yet. Source350/all207Adam restoration and initial fullstate serialization passed; full-data coverage, independent checkpoint restore and final goal remain incomplete.'
plan['updated_at_utc']=datetime.now(timezone.utc).isoformat();path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'receipt':rec(dest),'current_step':queue['published_step'],'DEV_TR_WER':[x['TR_WER'] for x in points],'new_all_six_winner':False}))
