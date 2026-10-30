"""Freeze ONE future metadata-only recovery observer; no execution."""
import ast,difflib,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
old=F/'seal_source30_saved163200_recovery_v1.py';new=F/'seal_source31_saved167600_recovery_v1.py';raw=old.read_bytes();t=raw.decode().replace('\r\n','\n');nl='\r\n'if b'\r\n'in raw else'\n';l=[]
def v(a,b,n=None):
 global t
 count=t.count(a);assert count>0 and(n is None or count==n),(a,count,n);l.append({'old_literal':a,'new_literal':b,'expected_count':count});t=t.replace(a,b)
v('163020_164620','167420_169020')
for a,b in [('163200','167600'),('163220','167620'),('163000','167400'),('163001','167401'),('163021','167437'),('163022','167438'),('329760922','338402719')]:v(a,b)
v('Reads ONLY21 new journal22','Reads ONLY37 new journal23')
v('invocations/owner_000022','invocations/owner_000023')
v('resident_RAM_training_actual_23','resident_RAM_training_actual_24')
v('source30','source31')
v('ROOT_source31_canonical_archive_saved167400_final_execution_review_v1.json','ROOT_source31_fresh_signature_collect_saved167400_final_execution_review_v1.json')
v('canonical_archive_resume','fresh_signature_resume')
v('commits_000022.jsonl','commits_000023.jsonl')
v('source29 published167400','source30 published167400')
v('New active22 only, historical21','New active23 only, historical22')
v('Original closed21 STAT','Original closed22 STAT')
v('new21','new37')
v('actual21','actual37')
v('original21_preserved_STAT_only','original22_preserved_STAT_only')
v('old21_historical_attempts','old22_historical_attempts')
v('no_old_whole_prefix21tail','no_old_whole_prefix22tail')
v("'new37_count':21", "'new37_count':37")
v("  'new_preparation_archive':record(checkpoint/'prepared_metadata.json.gz'),\n", "  'actual_signature_collect_observation_policy':review['fresh_metadata_signature_and_collect_observation_policy'],\n  'signature_collect_scope_admission':record(I/'actual_fresh_signature_collect_observation_admission.json'),\n")
ast.parse(t);inverse=t
for e in reversed(l):
 assert inverse.count(e['new_literal'])==e['expected_count'],e['new_literal'];inverse=inverse.replace(e['new_literal'],e['old_literal'])
assert inverse.replace('\n',nl).encode()==raw;assert not new.exists();new.write_bytes(t.replace('\n',nl).encode())
diff=F/(new.stem+'.diff')
with diff.open('x',encoding='utf-8',newline='\n')as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),t.splitlines(True),fromfile=str(old),tofile=str(new))))
ledger=F/'source31_natural167600_recovery_sealer_exact_raw_inverse_v1.json'
with ledger.open('x',encoding='utf-8',newline='\n')as f:json.dump({'old':pin(old),'new':pin(new),'replacements':l,'full_raw_inverse_exact':True,'original_newline_restored':repr(nl),'diff':pin(diff),'actual_execution_admitted':False},f,indent=2);f.write('\n')
print(json.dumps({'source':pin(new),'raw_inverse':pin(ledger)}))
