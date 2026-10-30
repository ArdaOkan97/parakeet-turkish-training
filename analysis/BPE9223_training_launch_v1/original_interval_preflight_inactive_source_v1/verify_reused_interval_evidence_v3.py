"""Finite metadata-only control using already closed real decoder observations."""
import copy,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
from fastmss_original_interval_preflight_v3 import specification,classify,require_covered,store_record,read_record,record,digest
N=Path(__file__).parent;F=N.parent/'resident_RAM_original_interval_post88426_failure_actual_v1';OUT=N/'CPU_evidence_actual_03'
def require(value,message):
 if not value:raise ValueError(message)
def write(path,value):
 with Path(path).open('x',encoding='utf-8',newline='')as f:json.dump(value,f,indent=2);f.write('\n')
require(not OUT.exists(),'Fresh E output');OUT.mkdir();(OUT/'records').mkdir()
require(sys.flags.utf8_mode==1 and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1','UTF8 and CUDA masked')
for key in('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):require(os.environ.get(key)=='1','Four thread caps1')
p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);who={'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()};began=time.monotonic();peak=0;samples=[]
def guard():
 global peak
 s=sample_memory(os.getpid(),'inactive_interval_evidence_CPU_control');m=s['system_memory'];w=s['worker_private_memory'];peak=max(peak,w['private_bytes'])
 require(s['paired_sampling_within_one_second'] and m['available'] and w['available'],'Actual paired telemetry')
 require(peak<=512*1024**2 and time.monotonic()-began<180,'Finite512MiB/180s')
 require(m['commit_available_bytes']>=2_000_000_000+256*1024**2 and m['physical_available_bytes']>=1_500_000_000+256*1024**2,'Full extra256MiB admission and unchanged headroom')
 require(shutil.disk_usage('E:/').free>=100_000_000_000+1024**2,'E100GB reserve plus bounded output')
 samples.append(s)
def read(path):return json.loads(Path(path).read_text())
negatives=[]
def rejects(name,call):
 try:call()
 except ValueError as error:negatives.append({'name':name,'rejected':True,'error':str(error)})
 else:raise AssertionError('Expected rejection: '+name)
try:
 guard();configuration=read(F/'production_configuration_v2/configuration.json');runtime_authority=configuration['CPU_template_configuration'];cert_path=F/'fourth_original_terminal_interval_geometry_certificate_v1.json';cert=read(cert_path)
 full=read(cert['independent_full_member_observation']['path']);probe=read(F/'probe_actual_01/report.json')
 failed=probe['results'][-1];a=failed['actual_decoder2'];decoder=a['frozen_reader_source']
 require(record(decoder['path'])==decoder and a['written']==360143 and a['count']==360144 and not a['passed'],'Genuine failed strict decoder result, unchanged decoder')
 require(full['native_interval_samples']==a['written'] and full['native_interval_f32_sha256']==cert['native_f32_sha256'] and full['full_source_range_equals_original_PTS_selected_PCM'],'Independent full-member exact real PCM evidence')
 spec=specification(cert['original_row'],cert['original_declaration'],cert['full_target_ids'],source=cert['original_TAR'],member=cert['recording_index_entry'],decoder_source=decoder,runtime_authority=runtime_authority)
 actual={'source':spec['source'],'member':spec['member'],'decoder_source':decoder,'written':a['written'],'count':a['count'],
  'original_decoder_returned':a['passed'],'strict_guards_passed':a['passed'],'original_error':a['error'],
  'native_f32_sha256':full['native_interval_f32_sha256'],'decoded_frames':a['decoded'],'evidence_reused_no_new_decode':True}
 entry=classify(spec,actual,evidence={'failed_original_decoder_report':record(F/'probe_actual_01/report.json'),
  'independent_full_member_report':record(cert['independent_full_member_observation']['path']),'certificate':record(cert_path)})
 require(entry['status']=='blocked_unresolved_original_interval' and entry['native_sample_shortfall']==1,'One-sample mismatch flagged, never tolerated')
 blocked_pin=store_record(OUT/'records',entry);require(read_record(blocked_pin,spec)==entry,'Exact persisted immutable observation')
 rejects('real_one_sample_short_original_interval_blocks_GPU',lambda:require_covered([entry]))
 # A real preceding covered row is reused from the SAME already closed probe.
 passed=next(item for item in probe['results']if item['actual_decoder2']['passed']);pa=passed['actual_decoder2'];info=pa['info']
 plan=read(probe['failed_update']['path']);row=plan['updates'][passed['update_index']]['rows'][passed['row_index']]
 tar=Path(r'E:\gpu_training_scratchpad')/'datasets/yodas3/audio'/(passed['row']['shard']+'.tar');st=tar.stat()
 pspec=specification(passed['row'],passed['declared_geometry'],row['tr_metadata_labels'],
  source={'path':str(tar),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'recorded_sha256':info['source_file_sha256']},
  member={'member_name':info['member_name'],'offset':info['member_offset'],'bytes':info['member_bytes']},decoder_source=pa['frozen_reader_source'],runtime_authority=runtime_authority)
 pact={'source':pspec['source'],'member':pspec['member'],'decoder_source':pspec['decoder_source'],'written':pa['written'],'count':pa['count'],
  'original_decoder_returned':True,'strict_guards_passed':True,'clean_f32_sha256':pa['PCM_f32_sha256'],'evidence_reused_no_new_decode':True}
 positive=classify(pspec,pact,evidence={'real_preceding_passed_probe':record(F/'probe_actual_01/report.json'),'row_index':passed['row_index'],'update_index':passed['update_index']})
 positive_pin=store_record(OUT/'records',positive);require(require_covered([read_record(positive_pin,pspec)]),'Real originally covered row accepted without changing geometry')
 for label,path,value in [('annotation',('row','end'),53.150),('target',('full_target_ids',0),1),('source_SHA',('source','recorded_sha256'),'0'*64),('member_offset',('member','offset'),1),('decoder_SHA',('decoder_source','sha256'),'0'*64),('runtime_authority_SHA',('runtime_authority','sha256'),'0'*64)]:
  altered=copy.deepcopy(spec);owner,key=path;altered[owner][key]=value
  rejects('cached_observation_invalidated_by_'+label,lambda changed=altered:read_record(blocked_pin,changed))
 bad=copy.deepcopy(actual);bad['member']['offset']+=1
 rejects('wrong_observed_member_rejected',lambda:classify(spec,bad,evidence={}))
 bad=copy.deepcopy(positive);bad['actual']['strict_guards_passed']=False
 rejects('covered_label_cannot_hide_failed_source_guard',lambda:require_covered([bad]))
 rejects('over16_row_implicit_batch_rejected',lambda:require_covered([positive]*17))
 rejects('empty_batch_rejected',lambda:require_covered([]))
 guard();require(not any(k in sys.modules for k in('torch','numpy','av')),'No audio/array/model/GPU runtime import')
 report={'kind':'ASR18_actual_reused_original_interval_preflight_evidence_CPU_control_v1','passed':True,'source':record(__file__),'preflight_source':record(S/'fastmss_original_interval_preflight_v3.py'),'producer':who,
  'closed_real_decoder_report':record(F/'probe_actual_01/report.json'),'independent_full_member_report':record(cert['independent_full_member_observation']['path']),
  'certificate':record(cert_path),'blocked_original_interval':blocked_pin,'covered_original_interval':positive_pin,'negative_controls':negatives,
  'new_audio_decodes':0,'new_GPU_model_or_training_calls':0,'sampler_journal_source_and_annotation_changes':0,'finite_evidence_only_not_live_integration':True,
  'separate_future_decoder_adapter_executed':False,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':peak,'resource_samples':samples}
 write(OUT/'report.json',report);print(json.dumps({'report':record(OUT/'report.json'),'passed':True,'new_audio_decodes':0,'negatives':len(negatives),'peak_private_bytes':peak}),flush=True)
finally:write(OUT/'CPU_completion.json',{'producer':who,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':peak,'actual_exit_unobserved_inside_child':True})
