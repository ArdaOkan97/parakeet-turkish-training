import ast,difflib,hashlib,json
from pathlib import Path
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');F=N/'resident_RAM_original_interval_post142894_failure_actual_v1'
oldpath=N/'resident_RAM_training_actual_16/seal_sixth_terminal_production_checkpoint_v2.py';old=oldpath.read_text(encoding='utf-8');new=old
changes=[('N=Path(__file__).parent;',"N=Path(r'E:\\gpu_training_scratchpad\\analysis\\BPE9223_training_launch_v1\\resident_RAM_training_actual_17');"),
 ('owner_000015','owner_000016'),('post140236_failure','post142894_failure'),('140400','143000'),('140237','142895'),('140238','142896'),('140201','142801'),('140200','142800'),('284977815','290088487'),('commits_000015','commits_000016'),
 ('sixth_original_terminal_interval_geometry_certificate_v1.json','seventh_original_full_clip_geometry_certificate_v1.json'),
 ('ROOT_sixth_original_terminal_interval_saved','ROOT_seventh_original_full_clip_saved'),('ROOT_sixth_original_terminal_interval_actual_CPU_review_v2.json','ROOT_seventh_original_full_clip_geometry_actual_CPU_review_v1.json'),
 ('same_pass_clean_evidence\'][4]','same_pass_clean_evidence\'][6]'),("metadata['rows'][4]","metadata['rows'][6]"),("'consumed_row_index':4","'consumed_row_index':6"),("'row_index':4","'row_index':6"),
 ('308496','231085'),('==241','==181'),('==121','==61'),('1929','1445'),('1928','1444'),('new37','new95'),('Actual37','Actual95'),("'owned_session':1903","'owned_session':47103"),
 ('healthy source19','healthy source20'),('sixth_terminal','seventh_full_clip'),('ASR18_actual_original_terminal_interval_corrected_and_saved','ASR18_actual_original_full_clip_corrected_and_saved')]
counts=[]
for a,b in changes:
 count=new.count(a);assert count>0,a;counts.append((a,b,count));new=new.replace(a,b)
ast.parse(new)
assert "same_pass_clean_evidence'][6]"in new and "metadata['rows'][6]"in new and "'consumed_row_index':6"in new and "'row_index':6"in new
assert 'range(142801,142896)'in new and '==231085'in new and '==181'in new and '==61'in new and '[1,1445,128]'in new
assert 'ROOT_seventh_original_full_clip_saved142800_execution_review_v1.json'in new
assert 'ROOT_seventh_original_full_clip_geometry_actual_CPU_review_v1.json'in new
path=F/'seal_seventh_full_clip_production_checkpoint_v1.py';assert not path.exists();path.write_bytes(new.replace('\n','\r\n').encode())
diff=F/'prior_sealer_to_seventh_current_binding.diff.txt';assert not diff.exists();diff.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=oldpath.name,tofile=path.name)),encoding='utf-8')
def pin(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
print(json.dumps({'sealer':pin(path),'delta':pin(diff),'execution':'NOT executed; requires genuine143000 fullpersistence/postpositive, native24824/currentCPU and liveowned47103. Reads exactly95newjournal events and the143000 preparedmetadataarchive, never oldwholejournal or tensorpayload.'}))