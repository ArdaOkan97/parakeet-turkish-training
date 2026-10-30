"""Prepare next genuine scheduled quality observer; never runs DEV or observer."""
import ast,difflib,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
old=F/'seal_periodic164000_source30_CPU11_helper3_max2_saved164200_v1.py';new=F/'seal_periodic168000_source31_CPU11_helper3_max2_saved168200_v1.py'
raw=old.read_bytes();text=raw.decode().replace('\r\n','\n');nl='\r\n'if b'\r\n'in raw else'\n';assert text.replace('\n',nl).encode()==raw;l=[]
for a,b in [('actual164k','actual168k'),('source30','source31'),('invocations/owner_000022','invocations/owner_000023'),('resident_RAM_training_actual_23','resident_RAM_training_actual_24'),
 ('164000','168000'),('164200','168200'),('160000','164000'),('72166','2901'),('1020','620')]:
 n=text.count(a);assert n>0,a;l.append({'old_literal':a,'new_literal':b,'expected_count':n});text=text.replace(a,b)
ast.parse(text);inv=text
for e in reversed(l):
 assert inv.count(e['new_literal'])==e['expected_count'],e['new_literal'];inv=inv.replace(e['new_literal'],e['old_literal'])
assert inv.replace('\n',nl).encode()==raw;assert not new.exists();new.write_bytes(text.replace('\n',nl).encode())
diff=F/(new.stem+'.diff')
with diff.open('x',encoding='utf-8',newline='\n')as f:f.write(''.join(difflib.unified_diff(old.read_text().splitlines(True),text.splitlines(True),fromfile=str(old),tofile=str(new))))
ledger=F/'source31_periodic168000_metadata_observer_exact_raw_inverse_v1.json'
with ledger.open('x',encoding='utf-8',newline='\n')as f:json.dump({'old':pin(old),'new':pin(new),'replacements':l,'full_raw_inverse_exact':True,'original_newline_restored':repr(nl),'diff':pin(diff),
 'actual_CPU_cohort':'167400 restored to168000 =600 native packets plus20 original same-plan unconsumed drain =620 acquires and returns',
 'execute_only_after_actual168000full350_normalclose0_preservation_resumedpositive_and_standard168200_postpositive':True,'no_extraDEV_or_job_admitted':True},f,indent=2);f.write('\n')
print(json.dumps({'source':pin(new),'ledger':pin(ledger)}))
