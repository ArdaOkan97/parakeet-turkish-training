"""Verify actual final native20 boundary, archive metadata, retire owned features."""
from pathlib import Path
import hashlib,json,os,sys
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from fastmss_feature_window_v1 import CACHE,cleanup_window,read_bound,record,write_new,scoped
from resource_watchdog_v2 import sample_memory
ROOT=Path(r'E:\gpu_training_scratchpad');A=ROOT/'analysis/ASR18_memory_diagnosis_v1/feature_window_native20_01'
F=ROOT/'analysis/FastMSS_integration_v1';RUN=F/'weak_fixed14_feature_window20_01'
def guard():
 v=sample_memory(os.getpid(),'CPU_metadata_owned_feature_retirement');w,s=v['worker_private_memory'],v['system_memory']
 assert w['available'] and s['available'] and v['paired_sampling_within_one_second'] and w['private_bytes']<2_000_000_000
 assert s['commit_available_bytes']>=2_000_000_000 and s['commit_used_bytes']/s['commit_limit_bytes']<.90
def main():
 report=json.loads((RUN/'report.json').read_bytes());owner=read_bound(report['owner'],guard)
 boundary_path=A/'final_coherent_feature_cleanup_boundary_v1.json'
 assert record(boundary_path,guard)['sha256']=='e7f38060300b5712eab7b77f3fce67d8bc49b20ca45c2441758bd77709078f8f'
 b=json.loads(boundary_path.read_bytes())
 native=read_bound(b['native_actual_closed'],guard);restore=read_bound(b['final_exact_restore'],guard)
 latest=read_bound(b['verified_final_coherent_latest_pointer'],guard);state=read_bound(b['checkpoint'],guard)
 assert native['actual_tool_exit_code']==native['actual_native_exit_code']==0 and native['owned_session']==38806
 assert native['native_PID']==23568 and native['actual_tool_chunk_id']=='b5e417' and native['native_updates']==20
 assert native['report']==b['native_report'] and native['checkpoint']==b['checkpoint'] and record(b['native_report']['path'],guard)==b['native_report']
 assert restore['passed'] and restore['updates']==20 and restore['checkpoint']==b['checkpoint']
 assert restore['all207_weights_all3_Adam_and_Python_NumPy_Torch_CUDA_RNG_exact'] and restore['actual_both_sampler_and_journal_prefix_verified']
 assert latest['committed_updates']==20 and latest['clocks']==[64270,2020,62270] and latest['writer_returned_after_fsync'] is True
 assert latest['checkpoint']==b['checkpoint'] and latest['payload']==state['payload']==b['checkpoint_payload']
 assert record(b['checkpoint_payload']['path'],guard)==b['checkpoint_payload']
 assert b['verified_all207_three_Adam_all_RNG_samplers_and_journal_prefix'] and b['serial_DEV_uses_checkpoint_not_features']
 snapshot=F/'clean_feature_fixture_original_bindings_v1.json'
 write_new(snapshot,{'kind':'FastMSS_fixture_original_metadata_snapshot_v1','owner':report['owner'],'owner_contents':owner,
  'actual_CPU_preparation_handoff':record(F/'FastMSS_feature_window20_actual_CPU_handoff_v1.json',guard),
  'retirement_boundary':record(boundary_path,guard),'copied_small_metadata_only':True,'native_sources_or_original_audio_modified':False})
 closures=[v['closure'] for v in report['updates']]
 result=cleanup_window(report['owner'],closures=closures,manifest_binding=report['manifest'],boundary=b,
  verify_boundary=lambda actual:actual==b,evidence_path=F/'weak_fixed14_feature_window20_durable_metadata_after_native20_v1.json',guard=guard)
 assert len(result['removed_exact_owned_files'])==682 and not scoped(report['owner']['path']).parent.exists() and not list(CACHE.iterdir())
 result.update(kind='FastMSS_actual_final_coherent_feature_window20_retirement_v1',passed=True,
  actual_native_owned_exit=native,verified_checkpoint_payload=b['checkpoint_payload'],verified_final_restore=b['final_exact_restore'],
  preserved_fixture_metadata=record(snapshot,guard),feature_files_only_deleted=True,original_audio_source_checkpoint_journal_untouched=True,
  native_qualification_or_production_approval=False)
 write_new(F/'weak_fixed14_feature_window20_actual_retirement_v1.json',result)
 print(json.dumps({'passed':True,'removed_exact_owned_files':len(result['removed_exact_owned_files']),
  'retirement':record(F/'weak_fixed14_feature_window20_actual_retirement_v1.json',guard),'fixture_metadata':record(snapshot,guard)}))
if __name__=='__main__':main()
