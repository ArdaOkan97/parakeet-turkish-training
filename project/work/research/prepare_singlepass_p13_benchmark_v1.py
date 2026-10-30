"""Read only actual p13/p12 archived metadata; no audio, model or scientific advance."""
import argparse,gzip,hashlib,json,lzma,sqlite3,struct,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from bpe_broad_audio_v1 import record,require
from fastmss_BPE9223_cpu_common_v12 import resource_guard as guard
ROOT=Path(r'E:\gpu_training_scratchpad')
def read(p):return json.loads(Path(p).read_bytes())
def checked(pin):
 guard();require(record(pin['path'])==pin,'Exact existing metadata authority');return read(pin['path'])
def put(p,raw):
 require(not p.exists(),'Fresh metadata output');p.write_bytes(raw);return record(p)
def main(a):
 c=read(a.configuration);review_pin=record(a.ROOT_review);review=checked(review_pin)
 run=Path(c['run_namespace']);out=Path(a.folder).resolve()
 require(out.is_relative_to(ROOT/'analysis') and not out.exists(),'Fresh E metadata fixture only');out.mkdir()
 db=sqlite3.connect((run/'checkpoints/catalog.sqlite').as_uri()+'?mode=ro',uri=True)
 try:
  p12=json.loads(db.execute('SELECT value FROM checkpoints WHERE id=?',('p000000012',)).fetchone()[0])
  p13=json.loads(db.execute('SELECT value FROM auxiliary WHERE id=?',('p000000013',)).fetchone()[0])
 finally:db.close()
 before={str(x):record(x) for x in [run/'checkpoints/state.json',run/'durable/latest.json',run/'control/worker_control_pool/registry.json',run/'journals/commits_000000.jsonl']}
 require(record(p12['archive']['path'])==p12['archive'],'Actual retired240 metadata archive')
 with gzip.open(p12['archive']['path'],'rb') as f:
  raw=f.read(4*1024**2);require(len(raw)<4*1024**2 and f.read(1)==b'','Bounded saved metadata')
 old=json.loads(raw);pointer=old['pointer'];raw=(json.dumps(pointer,ensure_ascii=False,indent=2)+'\n').encode()
 require(pointer['committed_updates']==240 and len(raw)==old['pointer_binding']['bytes'] and hashlib.sha256(raw).hexdigest()==old['pointer_binding']['sha256'],'Exact historical pointer240 byte reconstruction, not current260')
 pointer_pin=put(out/'actual_historical_p12_pointer_copy.json',raw)
 require(record(p13['archive']['path'])==p13['archive'],'Actual p13 immutable metadata archive')
 wanted={'coordinator_plan_copy.json','retained_feature_metadata.json'};selected={}
 with lzma.open(p13['archive']['path'],'rb') as f:
  require(f.read(len(b'ASR18_FUTURE_OWNED_AUXILIARY_1\n'))==b'ASR18_FUTURE_OWNED_AUXILIARY_1\n','Exact owned auxiliary format')
  size=struct.unpack('<Q',f.read(8))[0];require(0<size<=4*1024**2,'Bounded exact header');header=json.loads(f.read(size))
  require(header['manifest']['coordinator_plan']==p13['plan'],'Original p13 plan authority')
  for row in header['manifest']['files']:
   guard();pin=row['file'];count=struct.unpack('<Q',f.read(8))[0];require(count==pin['bytes'] and count<=16*1024**2,'Bounded metadata member')
   remaining=count;h=hashlib.sha256();parts=[];keep=Path(pin['path']).name in wanted
   while remaining:
    guard();b=f.read(min(1024**2,remaining));require(b,'Full archived member');h.update(b);remaining-=len(b)
    if keep:parts.append(b)
   require(h.hexdigest()==pin['sha256'],'Exact original member bytes')
   if keep:selected[Path(pin['path']).name]=(pin,b''.join(parts))
  require(f.read(1)==b'','Complete auxiliary archive EOF')
 require(set(selected)==wanted,'Actual p13 plan and complete640array metadata')
 plan_original,plan_raw=selected['coordinator_plan_copy.json'];plan=json.loads(plan_raw)
 require(plan['initial_committed_updates']==240 and len(plan['updates'])==20 and plan_original['sha256']==p13['plan']['sha256'],'Exact actual p13 241..260 plan')
 plan_pin=put(out/'actual_p13_plan_copy.json',plan_raw)
 golden_original,golden_raw=selected['retained_feature_metadata.json'];golden_pin=put(out/'actual_p13_golden_metadata_copy.json',golden_raw)
 template=checked(c['CPU_template_configuration']);source=template['source_contract_template'];numerical=checked(source['numerical_CPU_contract'])
 request={'kind':'ASR18_BPE9223_singlepass_CPU_request_v1','production_enabled':False,'prepared_is_committed':False,
  'plan':plan_pin,'session':c['session'],'DATA_evidence':c['planning_arguments']['DATA_evidence'],
  'numerical_CPU_contract':source['numerical_CPU_contract'],'augmentation_policy':source['augmentation_policy'],
  'augmentation_identity':source['augmentation_identity'],'original_augmentation_identity':source['original_augmentation_identity'],
  'frontend':numerical['frontend'],'ROOT_review':review_pin,'augmentation_ROOT_review':c['ROOT_CPU_execution_review'],
  'augmentation_configuration':c['CPU_template_configuration'],'worker_source':record(a.worker),'protocol_source':record(a.protocol),
  'parent_source':c['CPU_execution_delegate_source'],'run_namespace':str(run),'slot_index':0,'output_root':str(run/'provisional/slot_0'),
  'window_index':12,'window_id':'CPU_only_exact_p13_benchmark','provisional_predecessor':None,'current_checkpoint_pointer':pointer_pin}
 rp=put(out/'request.json',(json.dumps(request,ensure_ascii=False,indent=2)+'\n').encode())
 require(all(record(p)==pin for p,pin in before.items()),'All authoritative state/head/pool/journal unchanged')
 result={'kind':'ASR18_actual_p13_singlepass_benchmark_metadata_request_v1','request':rp,'configuration':record(a.configuration),
  'actual_original_plan':p13['plan'],'original_plan_copy':plan_original,'actual_historical_p12_archive':p12['archive'],
  'actual_historical_p12_pointer':old['pointer_binding'],'actual_p13_golden_metadata':golden_original,'golden_metadata_copy':golden_pin,
  'original_metadata_copied_byte_exact':True,'historical_pointer_never_current_restore_authority':True,
  'authoritative_state_pins_unchanged':before,'actual_optimizer_updates':0,'audio_GPU_or_model_used':False}
 result_pin=put(out/'handoff.json',(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode());print(json.dumps(result_pin))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--configuration',required=True);p.add_argument('--ROOT-review',dest='ROOT_review',required=True)
 p.add_argument('--worker',required=True);p.add_argument('--protocol',required=True);p.add_argument('--folder',required=True);main(p.parse_args())
