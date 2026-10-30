"""Focused CPU representation proof. Native-shaped records are EXPLICIT doubles."""
import argparse,copy,gc,gzip,hashlib,io,json,math,os,shutil,sys,time,weakref
from pathlib import Path
import psutil
F=Path(__file__).parent;P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(F)]
from bpe_broad_production_state_v1 import canonical,require
import asr18_canonical_prepared_archive_v1 as codec
from resource_watchdog_v2 import sample_memory
OUT=F/'canonical_archive_bytes_actual_CPU_01';BEGIN=time.monotonic();PEAK=0
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();require(len(b)<=8*1024**2,'Compact/source pin bound');return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def guard(admission=False):
 global PEAK
 s=sample_memory(os.getpid(),'canonical_archive_bytes_focused_CPU');m=s['system_memory'];PEAK=max(PEAK,s['worker_private_memory']['private_bytes']);extra=536870912 if admission else 0
 require(s['paired_sampling_within_one_second']and PEAK<536870912 and time.monotonic()-BEGIN<180
   and m['commit_available_bytes']>=2268435456+extra and m['physical_available_bytes']>=1768435456+extra
   and shutil.disk_usage('E:/').free>=100000000000,'Original512MiB/180s/fullprospective+prefetch/floors/E100GB');return s
def who(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
def stream_bytes(parts):
 out=io.BytesIO();codec.write_canonical_records(out,parts);return out.getvalue()
def tracked_graph(root):
 seen=set();todo=[root];count=0
 while todo:
  value=todo.pop()
  if id(value)in seen:continue
  seen.add(id(value));count+=int(gc.is_tracked(value))
  if type(value)in (dict,list,tuple):todo.extend(gc.get_referents(value))
 return {'reachable_identity_count':len(seen),'GC_tracked_reachable_objects':count}
def fixed_gzip(write):
 out=io.BytesIO()
 with gzip.GzipFile(filename='prepared_metadata.json',mode='wb',fileobj=out,mtime=0)as z:write(z)
 return out.getvalue()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--ROOT-review',required=True);a=parser.parse_args();me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'and os.environ.get('PYTHONUTF8')=='1'and all(os.environ.get(k)=='1'for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))and Path(os.environ['TORCH_EXTENSIONS_DIR']).is_relative_to(Path('E:/gpu_training_scratchpad')),'Separate E CPU-only environment')
 review=json.loads(Path(a.ROOT_review).read_text());require(review['accepted']is True and review['reviewer']=='/root'and review['scope']=='ASR18_canonical_bytes_prepared_archive_CPU_v1'and review['fixture_source']==pin(__file__)and Path(review['output']).resolve()==OUT.resolve(),'Exact focused ROOT scope')
 for p in review['input_pins']:require(pin(p['path'])==p,'Every exact compact/source input')
 require(not OUT.exists(),'One actual finite proof');initial=guard(True);OUT.mkdir();cases=[]
 def case(name,condition,**details):require(condition,name);cases.append({'name':name,'passed':True,**details})
 examples=[[],[{'binding':None,'metadata':{'unicode':'İğüş 中文 🙂','finite':[-0.,1.25,1e-17,True,False,None]},'native':{'routes':{0:{'x':1},1:{'x':2},10:{'x':3}}}}]]
 for n,records in enumerate(examples):case('exact_nested_original_canonical_'+str(n),stream_bytes([codec.encode_record(r)for r in records])==canonical(records))
 inp=json.loads((F/'exact_already_planned280_scalar_metadata_input_v1.json').read_text());plan=json.loads(Path(inp['source_plan']['path']).read_text());oracle=[r for u in plan['updates']for r in u['rows'][:14]]
 require(len(oracle)==280 and [r['ordinal']for r in oracle]==inp['original_ordered280_ordinals']and all(hashlib.sha256(canonical(r)).hexdigest()==b['canonical_original_plan_row_sha256']for r,b in zip(oracle,inp['original_ordered_rows'])),'Original280 row oracle unchanged')
 records=[];encoded=[];shared_policy={'kind':'METHOD_STUB_immutable_native_policy','native_math_performed':False}
 for i in range(200):
  u=copy.deepcopy(plan['updates'][i%20]);metadata={'update_index':i%20,'rows':u['rows'],'prepared_payload':{'state':'prepared_not_committed','rows':[{'source_id':r['source_id'],'language':r['language']}for r in u['rows']]}}
  native={'update':i+1,'phase':u['phase'],'mean_objective':1.25,'global_grad_norm':.5,'augmentation':metadata['prepared_payload'],'source_ids':[r['source_id']for r in u['rows']],'languages':[r['language']for r in u['rows']],'new_joint_policy':shared_policy,'joint_work_unit_allocation_routes':{j:{'kind':'METHOD_STUB_shape_only','T':r['encoder_frames'],'U':r.get('target_tokens',len(r.get('clean_target_ids',[])))}for j,r in enumerate(u['rows'])},'graph_boundary_memory_observations':[{'group':1,'seconds':.125,'collected_objects':7,'original_row_indices':list(range(len(u['rows'])))}]}
  record={'binding':{'kind':'METHOD_STUB_serialization_only','metadata_sha256':hashlib.sha256(canonical(metadata)).hexdigest()},'metadata':metadata,'native':native};records.append(record);encoded.append(codec.encode_record(record))
  if i in (0,19,199):case('exact_original_canonical_prefix_'+str(i+1),stream_bytes(encoded)==canonical(records))
  if i%20==0:guard()
 old=canonical(records);old_hash=hashlib.sha256(old).hexdigest();gz_old=fixed_gzip(lambda z:z.write(old));gz_new=fixed_gzip(lambda z:codec.write_canonical_records(z,encoded))
 case('complete200_original_JSON_bytes_exact',stream_bytes(encoded)==old,bytes=len(old),sha256=old_hash)
 case('fixed_header_gzip_compressed_bytes_exact',gz_old==gz_new)
 case('gzip_CRC_full_decoded_bytes_exact',gzip.decompress(gz_new)==old)
 target=OUT/'prepared_metadata.json.gz';codec.write_gzip_archive(target,encoded);case('actual_exclusive_checkpoint_gzip_decodes_to_exact_original',gzip.decompress(target.read_bytes())==old)
 try:codec.write_gzip_archive(target,encoded)
 except FileExistsError:case('original_exclusive_create_existing_path_failure',True)
 else:raise AssertionError('Exclusive create failed')
 old_graph=tracked_graph(records);new_graph=tracked_graph(encoded);case('bytes_GC_untracked_and_retained_graph_reduced',all(type(x)is bytes and not gc.is_tracked(x)for x in encoded)and new_graph['GC_tracked_reachable_objects']<old_graph['GC_tracked_reachable_objects'],old_graph=old_graph,new_graph=new_graph)
 class WeakList(list):pass
 witness=WeakList([{'x':[1,2,3]}]);ref=weakref.ref(witness);retained=[{'metadata':witness}];snapshot=[codec.encode_record(retained[0])];del witness;case('original_record_owner_retains_graph',ref()is not None);retained.clear();case('bytes_do_not_retain_original_graph',ref()is None,scope='Weakref-able list witness; production builtin lists are not weakref-able')
 cyclic=[];cyclic.append(cyclic)
 for name,value,error in [('circular',cyclic,ValueError),('nonfinite',float('nan'),ValueError),('unsupported',object(),TypeError)]:
  try:codec.encode_record(value)
  except error:case('original_canonical_'+name+'_rejection',True)
  else:raise AssertionError(name+' accepted')
 try:codec.write_canonical_records(io.BytesIO(),[b'{}',{'not':'bytes'}])
 except ValueError:case('writer_rejects_nonbytes_internal_miswire',True)
 else:raise AssertionError('Wrong type accepted')
 sentinel=OSError('METHOD_STUB_checkpoint_stream_write_failure')
 class BadStream:
  def write(self,data):raise sentinel
 try:codec.write_canonical_records(BadStream(),encoded)
 except OSError as error:case('stream_failure_original_exception_identity',error is sentinel)
 else:raise AssertionError('Stream failure swallowed')
 for name,bad in [('CRC_corruption',gz_new[:-8]+bytes([gz_new[-8]^1])+gz_new[-7:]),('truncation',gz_new[:-3])]:
  try:gzip.decompress(bad)
  except (OSError,EOFError):case('gzip_'+name+'_rejected',True)
  else:raise AssertionError(name+' accepted')
 case('record_order_change_detected',stream_bytes(encoded[::-1])!=old)
 shared_policy['deliberate_TEST_only_later_mutation']=True;case('later_mutable_alias_is_NEGATIVE_source_contract',stream_bytes(encoded)!=canonical(records),meaning='Such mutation is outside admitted source contract; not silently declared equivalent')
 shared_policy.pop('deliberate_TEST_only_later_mutation');case('shared_immutable_policy_snapshot_exact',stream_bytes(encoded)==canonical(records))
 for p in review['input_pins']:require(pin(p['path'])==p,'Every exact input unchanged')
 case('no_Torch_NumPy_or_model_runtime_import',not any(k in sys.modules for k in ('torch','numpy','transformers')))
 report={'kind':'ASR18_canonical_bytes_prepared_archive_actual_CPU_representation_proof_v1','passed':True,'identity':who(me),'launcher_identity':who(me.parent()),'ROOT_scope':pin(a.ROOT_review),'helper_source':pin(codec.__file__),'fixture_source':pin(__file__),'cases':cases,'case_count':len(cases),'complete_record_count':200,'uncompressed_original_bytes':len(old),'uncompressed_original_sha256':old_hash,'actual_gzip':pin(target),'old_reachable_graph':old_graph,'bytes_reachable_graph':new_graph,'input_pins_before_after_exact':True,'input_pins':review['input_pins'],'initial_resource_sample':initial,'last_resource_sample':guard(),'sampled_private_peak_bytes':PEAK,'wall_seconds':time.monotonic()-BEGIN,'native_receipt_fields':'EXPLICIT synthetic native-shaped scalars/route/boundary doubles; actual frozen original plan rows; NO actual optimizer step or scientific claim','scope_limits':['Pure CPU serialization/ownership proof, not measured production memory saving or speed.','Original native/GC/freeze/sync/optimizer/guard code is unchanged.','Snapshot equality requires source-bound no-later-mutation of archived records. Changed-alias negative is explicit.','Actual production validation must span>=1600 plus scheduled DEV boundary; no short-warm guarantee.'],'actual_GPU_native_audio_model_draw_operations':0}
 out=OUT/'report.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'report':pin(out),'cases':len(cases),'old_graph':old_graph,'bytes_graph':new_graph,'peak':PEAK}),flush=True)
if __name__=='__main__':main()
