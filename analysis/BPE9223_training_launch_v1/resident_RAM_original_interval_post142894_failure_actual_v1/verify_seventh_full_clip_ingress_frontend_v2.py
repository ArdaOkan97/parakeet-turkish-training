"""One real failed-row CPU ingress/frontend control, no training or GPU."""
import argparse,copy,hashlib,json,os,shutil,sys,time
from pathlib import Path
import psutil
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(P/'work/research')]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from fastmss_resident_feature_stream_v9 import _startup,AvailableMemoryPolicy
from fastmss_original_clip_geometry_overlay_v7 import match_original,decode_full_member
p_args=argparse.ArgumentParser();p_args.add_argument('--ROOT-review',required=True);args=p_args.parse_args();N=Path(__file__).parent;OUT=N/'ingress_frontend_actual_01';require(not OUT.exists(),'Fresh E actual control');OUT.mkdir()
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and sys.flags.utf8_mode==1,'CPU masked/upfront UTF8')
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):require(os.environ.get(key)=='1','Four CPU caps1')
p=psutil.Process();p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);who={'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()};began=time.monotonic()
class BoundedPolicy(AvailableMemoryPolicy):
 def __call__(self):
  require(time.monotonic()-began<180,'Finite180s CPU ingress control')
  require(shutil.disk_usage('E:/').free>=100_000_000_000,'E100GB reserve')
  sample=super().__call__();require(sample['system_memory']['commit_available_bytes']>=2000000000+268435456 and sample['system_memory']['physical_available_bytes']>=1500000000+268435456,'Original real headroom plus full extra256MiB on every CPU observation');return sample
policy=BoundedPolicy(2_000_000_000);guard=policy;reader=None;negative=[]
policy0=guard();require(policy0['system_memory']['commit_available_bytes']>=4_000_000_000+268435456 and policy0['system_memory']['physical_available_bytes']>=3_500_000_000+268435456,'Full2GB CPU observation admission plus original real headroom and full256MiB extra')
try:
 base=read_json(N.parent/'resident_RAM_original_interval_post140236_failure_actual_v1/production_configuration_v1/configuration.json')
 original=copy.deepcopy(base)
 base['original_clip_geometry_overlay']={'prior_overlay':copy.deepcopy(original['original_clip_geometry_overlay']),'source':record(S/'fastmss_original_clip_geometry_overlay_v7.py'),'certificate':record(N/'seventh_original_full_clip_geometry_certificate_v1.json'),'ROOT_review':record(args.ROOT_review)}
 base['prior_resident_RAM_original_interval_failure']=record(N/'actual_failure_closure.json')
 base['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=record(S/'fastmss_resident_feature_stream_v9.py')
 config=OUT/'configuration.json';write_json(config,base)
 unchanged=copy.deepcopy(base);unchanged.pop('original_clip_geometry_overlay');unchanged.pop('prior_resident_RAM_original_interval_failure');original.pop('original_clip_geometry_overlay');original.pop('prior_resident_RAM_original_interval_failure');unchanged['resident_RAM_resource_scheduling_policy']['CPU_protocol_source']=original['resident_RAM_resource_scheduling_policy']['CPU_protocol_source'];require(unchanged==original,'Only overlay/failure fields plus exact new CPU protocol source pin; resource values unchanged')
 core,single,source,validated,reader,augmenter,clean_reader,frontend=_startup(record(config),policy)
 plan=read_json(read_json(N/'remaining_current_plan_updates.json')['source_plan']['path'])
 normalized=core.bridge.normalize_updates(plan,population=validated['retained_rows'],guard=guard)
 update=normalized[14];expected=update['rows'][6];row=copy.deepcopy(update['coordinator_update']['rows'][6]);context=update['coordinator_update']['contexts'][6]
 row['audio_draw_context']=copy.deepcopy(context);row['augmentation_role']=expected['role']
 require(row['broad_original']['unique_id']=='alania:0ce0d55b73c946f80abea8bbe41f3b27_00000','Exact original failed ingress')
 wave,reference=clean_reader(row,expected,update['phase']);arrays=augmenter.frontend(frontend,wave)
 import numpy as np
 g=core.check_frontend(arrays,len(wave),np)
 require(g=={'clean16k_samples':231085,'valid_mel_frames':1444,'physical_feature_frames':1445,'encoder_frames':181}
  and reference['full_target_ids']==reader.seventh_certificate['full_target_ids'] and len(reference['full_target_ids'])==61
  and reference['expected_audio']['waveform_sha256']==reader.seventh_certificate['clean_f32_sha256'],'Full physical PCM/targets/features/mask')
 require(np.isfinite(arrays['input_features']).all() and arrays['attention_mask'].dtype==np.bool_ and int(arrays['attention_mask'].sum())==1444,'Actual finite original frontend normalization and full1444-valid/1445-physical mask')
 array_reports={k:{'shape':list(v.shape),'dtype':str(v.dtype),'array_sha256':hashlib.sha256(memoryview(v).cast('B')).hexdigest()} for k,v in arrays.items()}
 require(reference['before_draw_geometry_proof']['historical_mel_frames']==1444 and reference['before_draw_geometry_proof']['actual_physical_mel_frames']==1444,'Historical and physical mel1444 are explicit; full original samples remain intact')
 require(reference['before_draw_geometry_proof']['historical_declaration']['clean16k_samples']==231088 and reference['before_draw_geometry_proof']['actual_geometry']['clean16k_samples']==231085,'Explicit unchanged historical larger clean231088 versus all-physical clean231085; no padding/cropping')
 actual_reference=copy.deepcopy(reference);del arrays,wave
 clean_reader.finish_row();require(not reader.ledger.jobs and reader.active_job is None,'Actual raw wave released')
 certificate=reader.seventh_certificate;raw=copy.deepcopy(certificate['original_row']);d=copy.deepcopy(certificate['original_declaration'])
 def rejects(name,call):
  try:call()
  except ValueError as error:negative.append({'name':name,'rejected':True,'error':repr(error)})
  else:raise AssertionError('Expected genuine rejection: '+name)
 altered=copy.deepcopy(raw);altered['end']=14.442
 rejects('changed_original_annotation',lambda:match_original(certificate,altered,d,expected))
 altered_d=copy.deepcopy(d);altered_d['native_samples']-=1
 rejects('changed_historical_declaration',lambda:match_original(certificate,raw,altered_d,expected))
 altered_e=copy.deepcopy(expected);altered_e['full_target_ids'][0]=(altered_e['full_target_ids'][0]+1)%8192
 rejects('changed_full_target',lambda:match_original(certificate,raw,d,altered_e))
 altered_row=copy.deepcopy(raw);altered_row['ordinal']+=1
 rejects('changed_original_row',lambda:match_original(certificate,altered_row,d,expected))
 bad_source=copy.deepcopy(certificate);bad_source['original_TAR']['recorded_sha256']='0'*64
 rejects('changed_original_source_SHA',lambda:decode_full_member(reader.broad,raw,bad_source,guard=guard,channels_bound=8,frame_samples_bound=16384))
 bad=copy.deepcopy(certificate);bad['member_sha256']='0'*64
 rejects('changed_original_member_SHA',lambda:decode_full_member(reader.broad,raw,bad,guard=guard,channels_bound=8,frame_samples_bound=16384))
 from fastmss_raw_lifetime_v1 import raw_plan
 rp=raw_plan(native_samples=d['native_samples'],native_rate=48000,clean_samples=d['clean16k_samples'],phase=update['phase'],role='student_tr',channels_bound=8,frame_samples_bound=16384)
 policy.admit(rp['raw_peak_bytes']+rp['compressed_media_cap_bytes']);reader.ledger.begin('negative_EOF',rp)
 bad=copy.deepcopy(certificate);bad['actual_geometry']['native_samples']+=1
 with reader.ledger.stage('negative_EOF','decode_resample'):
  rejects('one_extra_physical_sample_must_fail_EOF',lambda:decode_full_member(reader.broad,raw,bad,guard=guard,channels_bound=8,frame_samples_bound=16384))
 reader.ledger.finish('negative_EOF');require(not reader.ledger.jobs,'Negative raw scope released')
 import fastmss_original_clip_geometry_overlay_v7 as seventh
 original_call=seventh.prior.OriginalRows.__call__;delegated=[];sentinel=object();prior_rows=[]
 policy6=base['original_clip_geometry_overlay']['prior_overlay']
 while policy6 is not None:
  cert6=core.read_bound(policy6['certificate'],guard);prior_rows.append({'ordinal':cert6['original_row']['ordinal'],'source_id':cert6['original_row']['unique_id']});policy6=policy6.get('prior_overlay')
 require(len(prior_rows)==6,'All exact six prior certificate dispatches')
 def delegation_probe(self,r,e,p):delegated.append((r,e,p));return sentinel
 try:
  seventh.prior.OriginalRows.__call__=delegation_probe
  for prior_row in prior_rows:
   prior_expected={}
   require(reader(prior_row,prior_expected,update['phase'])is sentinel and delegated[-1][0]is prior_row and delegated[-1][1]is prior_expected,'Exact prior6 route delegation, metadata-only call control')
  ordinary_row={'ordinal':-1,'source_id':'metadata_only_unmatched_route'};ordinary_expected={}
  require(reader(ordinary_row,ordinary_expected,update['phase'])is sentinel and delegated[-1][0]is ordinary_row and delegated[-1][1]is ordinary_expected,'Ordinary unmatched route delegates unchanged, metadata-only')
 finally:seventh.prior.OriginalRows.__call__=original_call
 require(base['original_clip_geometry_overlay']['prior_overlay']==read_json(N.parent/'resident_RAM_original_interval_post140236_failure_actual_v1/production_configuration_v1/configuration.json')['original_clip_geometry_overlay'],'Exact unchanged prior6 policy tree')
 bad_prior=copy.deepcopy(base['original_clip_geometry_overlay']['prior_overlay']['certificate']);bad_prior['sha256']='0'*64
 rejects('changed_prior6_certificate_pin',lambda:core.read_bound(bad_prior,guard))
 previous_policy_delegation={'passed':True,'metadata_only_call_route_control':True,'no_old_row_audio_replayed':True,'parent_method_restored':True,'prior6_policy_exact':True,'all_six_prior_row_routes':prior_rows,'ordinary_unmatched_route':True}
 reader.close();reader=None
 require(not __import__('torch').cuda.is_initialized(),'CPU-only frontend never initializes CUDA')
 report={'kind':'ASR18_actual_original_terminal_interval_corrected_ingress_frontend_control_v1','passed':True,'source':record(__file__),'producer':who,'configuration':record(config),'ROOT_source_review':base['original_clip_geometry_overlay']['ROOT_review'],'overlay_source':base['original_clip_geometry_overlay']['source'],'CPU_stream_source':record(S/'fastmss_resident_feature_stream_v9.py'),'certificate':base['original_clip_geometry_overlay']['certificate'],'failed_plan':read_json(N/'remaining_current_plan_updates.json')['source_plan'],'row_index':6,'update_index':14,'actual_geometry':g,'arrays':array_reports,'reference':actual_reference,'negative_controls':negative,'previous_policy_delegation':previous_policy_delegation,'actual_native_updates':0,'draws_consumed':0,'no_GPU_or_model':True,'no_source_index_original_plan_or_journal_mutation':True,'raw_buffers_released_and_metadata_handles_closed':True,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':policy.peak_private}
 write_json(OUT/'report.json',report);print(json.dumps({'report':record(OUT/'report.json'),'geometry':g,'negative_controls':negative,'peak_private_bytes':policy.peak_private}),flush=True)
finally:
 if reader is not None:reader.close()
 write_json(OUT/'CPU_completion.json',{'producer':who,'elapsed_seconds':time.monotonic()-began,'peak_private_bytes':policy.peak_private,'actual_exit_unobserved_inside_child':True})
