"""NEW source-only controller quota/current historical-winner/budget delta."""
from pathlib import Path
import ast,hashlib
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
old=S/'asr18_full_run_parent_controller_v2.py';new=S/'asr18_full_run_parent_controller_v3.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
def rep(a,b):
    global text
    assert text.count(a)==1,a[:120];text=text.replace(a,b)
rep("self.folder.mkdir();self.root.mkdir();self.control.mkdir();self.parent=ParentGuard(self.folder)",
"self.folder.mkdir();self.root.mkdir();self.control.mkdir();self.parent=ParentGuard(self.folder)\n        self.parent.configure_control_quota(self.control,c['control_log_policy'])")
rep("        write_json(path,value);return record(path)",
"        if path.is_relative_to(self.control):self.parent.control_guard(path,len(canonical(value))+4096)\n        write_json(path,value);return record(path)")
rep("        with log.open('ab') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())",
"        self.parent.control_guard(log,(log.stat().st_size if log.exists() else 0)+len(raw))\n        with log.open('ab') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())")
rep("        require(report is not None and artifact==self.current_capture['pointer'] and self.current_ack is not None,\n                'Only current actual recovered pointer can become a new qualified winner')",
"        require(report is not None and self.current_ack is not None,\n                'Actual independently sealed qualified winner, never a diagnostic candidate')")
rep("        require(score['strict_all_six_passed'] and source_request['identity']==self.identity,\n                'Exact integer all-six retention and original full-run identity')",
"        require(score['strict_all_six_passed'] and source_request['identity']==self.identity\n                and source_request['checkpoint_pointer']==artifact,\n                'Exact integer all-six retention and original immutable candidate/run identity')\n        if type(winner) is dict and 'artifact' in winner:\n            with self.parent.metadata_phase('read_index'):\n                with self.store.locked():\n                    state=self.store._load()[0]\n                    entries=[self.store._entry(key,verify_files=True) for key in state['winners']]\n                require(any(entry['pointer']==artifact and entry.get('qualification') for entry in entries),\n                        'Historical qualified winner remains actually registered/protected with its original immutable DEV')\n        else:\n            require(artifact==self.current_capture['pointer'],'Only current recovered candidate can be NEWLY promoted')")
rep("        self.native_sequence+=1;self.current_worker_nonce=secrets.token_hex(32)",
"        self.native_sequence+=1;self.current_worker_nonce=secrets.token_hex(32);self.native_started=time.monotonic()")
rep("    def run(self):\n        while True:",
"    def _normal_budget_barrier(self):\n        if self.gpu is None or self.current_capture is None:return False\n        # Whole-window budget boundary only: never interrupt Adam or shorten a\n        # pass. Preserve PRE/state/journal then actual0/current restore/prefix ACK.\n        reserve=(self.config['maximum_CPU_phase_seconds']+self.config['maximum_native_window_seconds']\n                 +self.config['maximum_archive_phase_seconds']+self.config['maximum_checkpoint_phase_seconds']\n                 +self.config['maximum_native_exit_seconds'])\n        require(0<reserve<self.config['maximum_GPU_child_wall_seconds'],\n                'Reviewed native wall ceiling must support one complete next window plus closure')\n        return time.monotonic()-self.native_started+reserve>=self.config['maximum_GPU_child_wall_seconds']\n\n    def run(self):\n        while True:")
rep("            action=self.durable.next_action()",
"            if self._normal_budget_barrier():\n                self._recover();continue\n            action=self.durable.next_action()")
ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
data=new.read_bytes();print({'path':str(new),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'AST_only':True})
