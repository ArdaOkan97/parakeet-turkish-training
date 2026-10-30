"""NEW finite two-window stop8, preserving normal >=4 weak storage cadence."""
from pathlib import Path
import ast,hashlib
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
old=S/'asr18_full_run_parent_controller_v5.py';new=S/'asr18_full_run_parent_controller_v6.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
text=text.replace("'finite_full_run_owner_four_update_pipeline_validation_v1'","'finite_full_run_owner_eight_update_two_window_pipeline_validation_v1'")
a="c['maximum_owner_validation_updates']==c['maximum_validation_updates_per_window']==4"
b="c['maximum_owner_validation_updates']==8 and c['maximum_validation_updates_per_window']==4"
assert text.count(a)==1;text=text.replace(a,b)
text=text.replace('ONE exact four-update owner validation','Exact original first8/two-window owner validation')
a="""            if self.config['owner_validation_only']:
                require(self.schedule.committed_updates()==4,'Bounded validation uses SAME original productionbody, exactstop4')
                self._recover();break"""
b="""            if self.config['owner_validation_only']:
                actual=self.schedule.committed_updates()
                require(actual in (4,8),'Same original productionbody, two actual complete4-update windows')
                if actual==8:
                    self._recover();break"""
assert text.count(a)==1;text=text.replace(a,b)
ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
raw=new.read_bytes();print({'path':str(new),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'AST_only':True})
old=R/'run_ASR18_full_run_owner_v7.py';new=R/'run_ASR18_full_run_owner_v8.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig').replace('from asr18_full_run_parent_controller_v5 import run','from asr18_full_run_parent_controller_v6 import run')
ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
raw=new.read_bytes();print({'path':str(new),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'AST_only':True})
