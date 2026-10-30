"""Compare actual single-pass CPU0 to the original320-row archived window."""
import hashlib,json,sys,time
from pathlib import Path
P=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
import fastmss_singlepass_CPU_v1 as protocol
import fastmss_BPE9223_cpu_common_v12 as core
from fastmss_owned_process_closure_v2 import verify
from atomic_io_retry_v1 import write_json

def main():
 n=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1')
 m=n/'singlepass_CPU_p13_benchmark_metadata_v2';a=n/'singlepass_CPU_p13_benchmark_actual_v1'
 def read(p):return json.loads(Path(p).read_bytes())
 h=read(m/'handoff.json');request=read(m/'request.json');report=read(a/'report.json')
 actual=read(a/'actual_owned_process0.json')
 verify(actual,kind='ASR18_owned_CPU_process_completion_v1',owner='/root/memory_diagnosis',report=protocol.record(a/'report.json',core.resource_guard),guard=core.resource_guard)
 assert report['passed'] is True and report['full_rows']==report['actual_decode_count']==320
 assert report['actual_optimizer_updates']==report['draws_consumed']==0
 assert actual['worker_source']==report['worker_source']==request['worker_source']
 assert actual['parent_source']==report['parent_source']==request['parent_source']
 golden=read(h['golden_metadata_copy']['path'])['metadata'];arrays=rows=payloads=0;seconds=0
 keys=('source_id','language','context','role','original_row_sha256','raw_text_sha256','cleaner_original_ordinal','full_target_ids','target_ids_u32_sha256','target_tokens','encoder_frames','sample_rate','samples','clean_PCM_sha256','prepared_PCM_sha256')
 for i,part in enumerate(report['updates']):
  old=golden['update_%02d.json'%i];new=protocol.read_bound(part['update'],core.resource_guard)
  assert len(old['rows'])==len(new['rows'])==16
  assert new['prepared_payload']==old['prepared_payload'];payloads+=1
  for x,y in zip(old['rows'],new['rows']):
   assert all(x[k]==y[k] for k in keys);rows+=1;seconds+=y['samples']/16000
   for key in ('input_features','attention_mask'):
    left,right=x[key],y[key]
    assert all(left[k]==right[k] for k in ('shape','dtype','array_sha256'))
    assert all(left['file'][k]==right['file'][k] for k in ('bytes','sha256'))
    assert protocol.record(right['file']['path'],core.resource_guard)==right['file'];arrays+=1
 assert rows==320 and arrays==640 and payloads==20
 assert all(protocol.record(p,core.resource_guard)==pin for p,pin in h['authoritative_state_pins_unchanged'].items())
 observed=read(a/'ROOT_observed_actual_CPU0.json');family=read(a/'owned_CPU_family_completion.json')
 result={'kind':'ASR18_actual_singlepass_p13_golden_CPU_benchmark_v1','passed':True,'source':protocol.record(__file__,core.resource_guard),
  'CPU_process0':protocol.record(a/'actual_owned_process0.json',core.resource_guard),'report':protocol.record(a/'report.json',core.resource_guard),
  'actual_parent_and_child_wall_seconds':observed['observed_wall_seconds'],'worker_wall_seconds':report['wall_seconds'],
  'old_CPU_worker_wall_seconds':90.312,'CPU_speedup':90.312/report['wall_seconds'],
  'audio_seconds':seconds,'full_rows':rows,'full_file_and_array_SHA_identical':arrays,'full_PCM_labels_geometry_identical':rows,
  'exact_augmentation_payloads':payloads,'timings':report['timings'],'family':family,
  'authoritative_state_journal_head_pool_unchanged':True,'actual_native_updates':0,'draws_consumed':0,
  'end_to_end_overlap_or_training_throughput_proved':False,'transient_fsync_removed':True,'checkpoint_durability_unchanged':True}
 out=a/'actual_golden_CPU_summary_v1.json';assert not out.exists();write_json(out,result)
 print(json.dumps({'summary':protocol.record(out,core.resource_guard),'wall_seconds':report['wall_seconds'],'CPU_speedup':result['CPU_speedup'],'array_matches':arrays,'payload_matches':payloads}),flush=True)

if __name__=='__main__':main()
