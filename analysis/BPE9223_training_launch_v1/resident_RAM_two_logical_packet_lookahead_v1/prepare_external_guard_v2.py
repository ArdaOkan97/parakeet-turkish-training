import ast,difflib,hashlib,json
from pathlib import Path
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1')
old=F.parent/'resident_RAM_native_pressure_timeout156537_failure_actual_v1/inactive_packet_credit_v1/launch_one_reviewed_packet_credit_CPU_v1.py'
text=old.read_text(encoding='utf-8-sig');edits=[]
def edit(a,b,n=1):
 global text
 assert text.count(a)==n,(a[:60],text.count(a),n)
 text=text.replace(a,b);edits.append({'old_literal':a,'new_literal':b,'expected_count':n})
edit("review=json.loads(Path(a.ROOT_review).read_text());inputs={k:review[k]for k in ['fixture_source','helper_source','CPU_source','quiescence_helper','inactive_driver_source','inactive_wrapper_source']}","review=json.loads(Path(a.ROOT_review).read_text());inputs=review['input_pins']\n require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and os.environ.get('PYTHONUTF8')=='1' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'Original CPU-only process-local masks')\n require(Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Applied E runtime including extensions')\n require(review['fixture_source']==pin(F/'verify_two_logical_packet_one_receiver_CPU_v1.py') and review['external_launcher_source']==pin(__file__),'Exact frozen fixture/external guard scope')")
edit("for key,item in inputs.items():require(pin(item['path'])==item,'Exact canonical input pin '+key)","for item in inputs:require(pin(item['path'])==item,'Exact canonical input pin '+item['path'])")
edit("output=F/'actual_CPU_01';require(not output.exists()and not(F/'actual_CPU_01_external_result.json').exists(),'Exactly one fresh actual control')", "output=F/'finite_transport_actual_CPU_01';require(not output.exists()and not(F/'finite_transport_actual_CPU_01_external_result.json').exists(),'Exactly one fresh actual control')")
edit("actual_CPU_01_", "finite_transport_actual_CPU_01_",9)
# Previous output replacement already includes new name, avoid duplicated prefix from generic replacement.
assert 'finite_transport_finite_transport' in text
edit('finite_transport_finite_transport','finite_transport')
edit("inputs['fixture_source']['path']","review['fixture_source']['path']")
edit("for key,item in inputs.items():require(pin(item['path'])==item,'Every frozen input unchanged '+key)","for item in inputs:require(pin(item['path'])==item,'Every frozen input unchanged '+item['path'])")
edit("deaths=[{'identity':old,'exact_identity_dead':not same(old)}for old in known.values()]", "if cleanup and error is None:error='Owned family required cleanup after fixture result; not a clean finite pass'\n deaths=[{'identity':old,'exact_identity_dead':not same(old)}for old in known.values()]")
ast.parse(text)
path=F/'launch_one_reviewed_two_logical_packet_CPU_v1.py';assert not path.exists();path.write_bytes(text.replace('\n','\r\n').encode())
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
ledger={'old_source':pin(old),'new_source':pin(path),'edits_on_LF_normalized_text':edits,
 'newlines':'original LF-normalized for edits then emitted CRLF','scope':'INACTIVE source preparation, no actual control executed'}
with (F/'external_guard_source_edit_ledger_v1.json').open('x',encoding='utf-8')as f:json.dump(ledger,f,indent=2);f.write('\n')
plan={'kind':'ASR18_INACTIVE_two_logical_packet_external_guard_launch_plan_v1','actual_control_executed':False,
 'external_launcher_source':pin(path),'fixture_source':pin(F/'verify_two_logical_packet_one_receiver_CPU_v1.py'),
 'external_argv':['E:/gpu_training_scratchpad/scratch/BPE_broad_CPU_preflight_env_v2/Scripts/python.exe','-Xutf8','-B',str(path),'--ROOT-review','<ROOT exact scope receipt>'],
 'working_directory':'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20',
 'storage_prerequisite':['Read E:/gpu_training_scratchpad/storage_config.json','Dot-source E:/gpu_training_scratchpad/use_storage.ps1 INCLUDING TORCH_EXTENSIONS_DIR before the external Popen'],
 'process_local_environment':{'CUDA_VISIBLE_DEVICES':'-1','PYTHONUTF8':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'},
 'priority':'BelowNormal for external fixture and all synthetic owned roles','internal_seconds':180,'external_seconds':240,
 'sampled_family_private_bytes':536870912,'full_prospective_family_bytes':536870912,'full_additional_prefetch_bytes':268435456,
 'commit_at_admission_bytes':2805306368,'physical_at_admission_bytes':2305306368,
 'every_sample_commit_bytes':2268435456,'every_sample_physical_bytes':1768435456,'E_free_bytes':100000000000,
 'fresh_output':str(F/'finite_transport_actual_CPU_01'),'fresh_external_records_prefix':str(F/'finite_transport_actual_CPU_01'),
 'shutdown':'Record all observed PID/create-time pairs; real Popen wait; only exact owned descendants cleanup; any required cleanup prevents clean-pass claim',
 'running_production_env_or_sources_changed':False}
# full512MiB +268435456 +2GB/1.5GB is 2,805,306,368 and2,305,306,368 exactly.
with (F/'inactive_external_guard_launch_plan_v1.json').open('x',encoding='utf-8')as f:json.dump(plan,f,indent=2);f.write('\n')
print(json.dumps({'external_launcher':pin(path),'launch_plan':pin(F/'inactive_external_guard_launch_plan_v1.json'),'AST_parse_only':True}))
