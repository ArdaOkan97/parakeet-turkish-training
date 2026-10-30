"""CPU-only prospective experiment plan; does not run any training."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/parakeet_turkish'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def ref(name):
    p=OUTPUT/name
    return {'path':str(p),'sha256':sha(p),'exists':True} if p.exists() else {'path':str(p),'exists':False}
def read(name):return json.loads((OUTPUT/name).read_text(encoding='utf-8'))

data=read('tr_public_v2_preparation.json')
pool=read('runs/adaptive_20261010_continuation_02_resume_B8/training_pool.json')
groups=defaultdict(list)
for row in pool['replay']:groups[row['language']].append(row['duration'])
replay_means={language:sum(values)/len(values) for language,values in groups.items()}
replay_mean=sum(replay_means.values())/len(replay_means)
n=data['CPU_tokenizer_accepted_clips'];hours=data['CPU_tokenizer_accepted_hours']
quarter=math.ceil(n*.25/8)
tr_mean=hours*3600/n
quarter_audio_seconds=quarter*8*(tr_mean+replay_mean)
probe=read('cuda_cache_reclaim_gpu_probe_v4.json')
rate=probe['branches']['reclaim']['actual_audio_min_per_wall_min']
estimate_seconds=quarter_audio_seconds/rate

plan={
 'schema':1,'kind':'prospective_data_rank_native_objective_controller4_plan_v1',
 'status':'CPU_plan_only_no_new_controller_or_training_launched','created_at_utc':datetime.now(timezone.utc).isoformat(),
 'current_run_protection':{'controller03':ref('run_experiment_loop_continuation_v3.py'),
   'action':'Current1821 exact-state B8 continuation remains unchanged; no parallel GPU job or live source modification'},
 'quality_objective':{'turkish_DEV_wer_target':.1193939393939394,'strong_DEV_wer':.0993939393939394,
   'required_retention':'All five fixed DEV language gates, individually',
   'selection_split':'Fixed FLEURS DEV300 only','TEST_policy':'No TEST access for data/rank/recipe/LR/stop decisions',
   'persistent_goal':'Finite windows are recovery/evaluation intervals; resource/deadline stop checkpoints and leaves objective incomplete'},
 'expanded_data':{'manifest_path':data['CPU_tokenizer_accepted_manifest_path'],
   'manifest_sha256':data['CPU_tokenizer_accepted_manifest_sha256'],'CPU_eligible_clips':n,'CPU_eligible_PCM_hours':hours,
   'sources':data['CPU_tokenizer_accepted_source_counts'],'max_target_tokens':180,
   'preparation':ref('tr_public_v2_preparation.json'),'exact_labels':ref('tr_public_v2_tokenizer_acceptance.json'),
   'status':'CPU tokenizer/isolation prepared; independent fullWAV audit and actual frozen-teacher cache acceptance required before training',
   'sampler_policy':'New accepted source pool requires fresh Adam/sampler warmstart, never an exact resume from the old pool',
   'frontend_policy':'Keep the current pinned frontend for the first data/rank comparisons; any native frontend change needs a separate audited cache namespace'},
 'common_runtime':{'physical_encoder_batch':8,'accumulation':2,'effective_batch':16,
   'replay_clip_fraction':.5,'replay_language_sampling':'Uniform draw over en/es/it/fr/de, deterministic reservoir/cycle sampler',
   'allocator_cap_gb':9,'optimizer_end_cache_cleanup_threshold_gb':4,'actual_policy_proof':ref('memory_policy_launch_validation_v1.json'),
   'critical_commit_fraction':.95,'critical_headroom_bytes':2_000_000_000,'moderate_commit_fraction':.9,
   'moderate_observations':3,'fresh_interval_seconds':5,'no_silent_B1_fallback':True,
   'training_coefficients':{'token_KL':5,'duration_KL':.5,'feature_distillation':1},
   'precision':'BF16 autocast, FP32 adapters/heads/Adam; backend flags explicitly recorded per new worker',
   'warmup_updates':25,'warmup_presentations':400,'head_lr':5e-5,'lora_lr':3e-4,'new_row_multiplier':6,
   'historical_full_BF16_original_vs_v2_equivalence_gate_passed':False},
 'arm_design':[
   {'name':'expanded_data_rank8_control','priority':1,'start':'Immutable latest qualifying controller03 DEV-best weights',
    'pool':'New53.22h accepted pool plus identical replay pool','rank':8,'alpha':16,
    'optimizer':'Fresh AdamW/sampler; declared warmstart, no exact-state claim',
    'objective':'Current verified reduced TDT sigma0/omega0, equal length-normalized clip mean',
    'role':'Establish expanded-data baseline before changing rank or the native recipe'},
   {'name':'expanded_data_rank16','priority':2,'start':'Same frozen pre-arm checkpoint as rank8 control, upgraded function-preservingly',
    'pool':'Same accepted pool and ordered source draws','rank':16,'alpha':32,'optimizer':'Fresh AdamW/sampler matched to rank8 control',
    'objective':'Same as control','GPU_readiness':'Root-only actual initialization/loss/gradients/features/transcripts/memory comparison must precede this arm',
    'role':'Measure adapter-capacity effect at identical TR/replay presentations, not against an Adam-continuation control'},
   {'name':'native_kernel_same_objective','priority':3,'start':'Same common checkpoint, data and rank as a paired control',
    'objective':'Actual pinned native kernel sigma0/omega0/reductionnone then existing per-clip normalization',
    'prerequisites':'Native imports and root soloGPU same-logits forward/all-gradient tests, blank/duration ordering and memory profile must pass',
    'role':'Implementation fidelity comparison; any performance measure must use a solo GPU'},
   {'name':'released_native_objective','priority':4,'start':'New declared DEV-best warmstart after native implementation readiness',
    'objective':{'sigma':.02,'omega':.1,'reduction':'mean_volume'},
    'preservation':'Keep multilingual replay/KD/feature terms and individual DEV retention gates; record scale of each component',
    'optimizer':'First retain matched LR/Adam settings to isolate objective; native betas(.9,.98),decay.001,cosine is a later separately declared recipe arm',
    'role':'Different scientific objective/recipe, never claim bitwise equivalence to current sigma0 equal-clip training'}],
 'native_readiness':{'source_review':ref('nemo_training_review_v1.json'),
   'official_review_commit':'e4a34fd25f895ecd7f64f0e56bc288165e078da9',
   'native_install':'Root isolated Windows install underway; native imports/CUDA TDT not yet demonstrated when this plan was written',
   'executed_reference_scope':'Nine extracted official PyTorch recurrence FP32CPU cases, two allocation-only device edits; not actual native CUDA/full-model proof',
   'current_oracle_scope':'Earlier original/v2 oracles were pinned Transformers TDT, not executed NeMo loss',
   'blank_and_vocab':{'HF_token_rows':8199,'HF_blank':8192,'HF_new_char_rows':[8193,8194,8195,8196,8197,8198],
      'duration_rows':[8199,8200,8201,8202,8203],
      'preferred_bridge':'Keep student/teacher/tokenizer/decoder/KD in original HF IDs; explicitly wire nonlast blank into native loss if supported',
      'if_native_requires_blank_last':'Loss-only token-row permutation: old lexical0:8192 unchanged; newchars8193:8199->8192:8198; blank8192->8198; duration rows unchanged. Map native loss labels only and backprop through verified permutation.',
      'full_native_migration':'Separate full weight/ID/tokenizer/teacher/KD/export mapping audit; generic vocabulary replacement must not reset trained prediction or joint modules'},
   'reduction_detail':'Current objective mean(loss_i/U_i) differs from native sum(loss_i)/sum(U_i). For native mean_volume use the full effective16 target-token denominator across both microbatches, not an average of two per-microbatch volumes.',
   'sigma_omega_detail':'Native omega mixture may alter auxiliary RNG behavior; include kernel choice/sigma/omega state in provenance and matching, never label native Lightning fresh training as custom Adam exact resume.',
   'memory_detail':'Native full-vocabulary joint logits/gradients can dominate memory. Profile longest real TRAIN shapes at9GB cap, use declared native fused-joint subbatching/recomputation if needed while keeping physical encoder B8, and measure actual end-to-end throughput. No capacity claim based on the current reduced-loss3GB footprint.',
   'other_recipe_changes':'Native dither/SpecAug/cosine/Adam/frontend differences are independent axes; do not combine them silently with rank/data changes'},
 'rank16':{'plan':ref('rank16_ablation_plan_v1.json'),'upgrade_helper':ref('upgrade_lora_rank_v1.py'),
   'actual_CPU_proof':ref('rank16_upgrade_actual_cpu_verification_v1.json'),
   'initialization':'Scale2 preserved; old A rows/B columns copied, randomnewA+zero newB; heads unchanged; fresh optimizer only',
   'proof_limit':'96 actual trained projections checked onCPU, not full-encoder/CUDA equivalence',
   'rank8_adapter_parameters':1572864,'rank16_adapter_parameters':3145728},
 'matched_exposure_and_budget':{'actual_pool_count_is_authoritative_after_cache':True,'TR_draws_per_update':8,
   'quarter_epoch_updates_if_all_CPU_eligible_accepted':quarter,'quarter_TR_presentations':quarter*8,'quarter_replay_presentations':quarter*8,
   'TR_mean_seconds_CPU_pool':tr_mean,'replay_mean_seconds_uniform_language_draw':replay_mean,
   'replay_per_language_mean_seconds':replay_means,'estimated_quarter_total_audio_hours':quarter_audio_seconds/3600,
   'planning_only_reference_TRAIN_probe_rate_audio_min_per_wall_min':rate,
   'estimated_quarter_training_wall_minutes_at_that_rate':estimate_seconds/60,
   'estimate_scope':'Arithmetic estimate using source1821 100-update measured rate and actual pool duration means; new-data cache/startup, native loss, long-clip distribution and DEV overhead unmeasured',
   'policy':'A3h interval cannot promise all four arms plus new cache. First complete readiness/new-cache and rank8 expanded arm; compare rank16 at identical exposure only when time permits, otherwise carry immutable common-start and measured exposure into next interval. Do not select short partial trials as equal-exposure evidence.'},
 'controller4_contract':{'version':'NEW files only; no modifications to frozen03/helpers/reports',
   'initial_inputs':'Owned03 exit receipt, immutable common DEV-best snapshot, complete newWAV/label/cache audits, new trainer/launcher/math source hashes and soloGPU readiness records',
   'state_fields':['objective_complete=false until competitive verified','parent loop/run/selected step','manifest/pool/cache/tokenizer hashes','rank/objective/backend flags','warmstart vs exactresume mode','ordered source draw hash','TR/replay presentations and audio seconds','actual update-only wall rate','DEV validation and five gates','worker self identity/private/system commit','resource stop reason/resume checkpoint','new finite deadline and next action'],
   'candidate_selection':'DEV-only Turkish WER subject to each retention gate; record WER,CER,new-char emission/coverage,word-boundary errors and duration diagnostics',
   'minimum_evaluation':'Initial300DEV plus quarter-exposure checkpoints, retention stopping after two consecutive failures and plateau policy declared before launch',
   'final_publication':'After a winner is frozen, merge/export/reload ordinaryHF model and recheck mergedDEV; fresh isolated6language TEST/report/audits only for final reporting, never tuning. Preserve all previous exports/reports and reuse audited fixed strong TEST read-only.',
   'launch_authority':'Root alone after review/freeze and no GPU overlap'}
}
for name in ('adaptive_controller4_plan_v1.json','adaptive_controller4_plan_v1.txt'):
    if (OUTPUT/name).exists():raise FileExistsError(name)
(OUTPUT/'adaptive_controller4_plan_v1.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=f'''Controller4 prospective plan (CPU only; current03 unchanged)

Expanded data: {n:,} CPU-eligible clips / {hours:.4f} PCM hours. FullWAV audit and actual teacher-cache acceptance remain required. New pool means fresh Adam/sampler warmstart, not exact old-pool resume.

First establish expanded-data rank8 control. Compare rank16 from the same frozen common checkpoint with a fresh optimizer, identical source draws and equal exposure. Rank16 preserves scale2 via copied oldA/B, randomnewA+zero newB. Actual96-projection CPU proof exists; root soloGPU full-runtime readiness remains required.

Native implementation must first import and execute pinned NeMo CUDA TDT on identical actual logits/labels, with blank/duration ordering and all-gradient/memory proof. The current original/v2 oracle was Transformers. The executed new NeMo reference was extracted FP32CPU recurrence, not native CUDA.

Keep a native same-objective control (sigma0/omega0/equal-clip normalization) separate from released recipe sigma.02/omega.1/mean_volume. Mean_volume needs one effective16 token denominator. Keep HF tokenizer/teacher/KD indices intact where possible; generic vocabulary replacement must not reset learned heads. Native full-joint memory may require explicitly measured fused-joint subbatching, while encoder physicalB8 remains unchanged.

A quarter expanded epoch is about {quarter:,} updates / {quarter*8:,} TR plus {quarter*8:,} replay presentations if allCPU candidates pass cache. Duration-mean arithmetic estimates {quarter_audio_seconds/3600:.2f} audio hours and {estimate_seconds/60:.1f} training minutes at the prior100-update rate; new caching, native loss and DEV overhead are excluded/unmeasured. A3h window cannot promise all arms. Preserve matched common-start/exposure for another interval if needed.

Use verified9GB cap/4GB update-end cleanup, actual worker identity and95%/2GB immediate plus sustained90% guards. All candidate choices remain fixedDEV-only with five retention gates. Window/resource stop saves state and keeps the competitive-WER objective active. Root alone launches any GPU work.
'''
(OUTPUT/'adaptive_controller4_plan_v1.txt').write_text(text,encoding='utf-8')
print(json.dumps({'saved':str(OUTPUT/'adaptive_controller4_plan_v1.json'),'quarter_updates':quarter,'estimated_training_minutes':estimate_seconds/60}))
