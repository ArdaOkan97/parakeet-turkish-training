"""NEW source-only AUX boundary and bounded control/stdout quota successor."""
from pathlib import Path
import ast,hashlib
P=Path(__file__).resolve().parents[2];R=P/'work/research'
old=R/'run_ASR18_full_run_owner_v3.py';new=R/'run_ASR18_full_run_owner_v4.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
def rep(a,b):
    global text
    assert text.count(a)==1,a[:120];text=text.replace(a,b)
rep('class ParentGuard:', '''CONTROL_POLICY={'kind':'ASR18_full_run_nonjournal_control_log_quota_v1',
    'allocation_cluster_bytes':524288,'total_allocated_cap_bytes':4294967296,
    'fixed_control_allocation_cap_bytes':1073741824,'maximum_GPU_worker_directories':48,
    'reserved_each_GPU_worker_allocation_bytes':67108864,'maximum_worker_metadata_files':32,
    'maximum_worker_directories':8,'ordinary_metadata_file_cap_bytes':1048576,
    'stage_log_cap_bytes':8388608,'stdout_log_cap_bytes':8388608,
    'prospective_atomic_replace_reserve_bytes':2097152}

def allocated(size):
    return ((max(int(size),1)+524287)//524288)*524288

def worker_control_guard(folder):
    # Only fixed startup/footer/current IPC files live here. Every native row
    # stage and window ACK/boundary lives in the separately reserved AUX tree.
    total=0;files=0;directories=0
    for root,dirs,names in os.walk(folder):
        for name in dirs:
            path=Path(root)/name;require(not path.is_symlink(),'No control directory aliases')
            directories+=1;total+=524288
        require(directories<=CONTROL_POLICY['maximum_worker_directories'],'Bounded worker control directories')
        for name in names:
            path=Path(root)/name;require(path.is_file() and not path.is_symlink(),'Only exact regular worker control files')
            require(path.suffix.lower() in ('.json','.jsonl','.log','.txt','.tmp'),
                    'No tensor/audio/model/unknown files in control-log domain')
            size=path.stat().st_size;limit=(8388608 if name in ('stages.jsonl','worker.log') else 1048576)
            require(size<=limit,'Source-declared per-file control bound')
            files+=1;total+=allocated(size)
        require(files<=CONTROL_POLICY['maximum_worker_metadata_files'],'Bounded source control file count')
    require(total+CONTROL_POLICY['prospective_atomic_replace_reserve_bytes']<=67108864,
            'Cluster-rounded entire worker control reservation, including transient replacement')
    return total

class ParentGuard:''')
rep("self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start();deadline=time.monotonic()+3",
"self.control=None;self.worker_slots=[]\n        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start();deadline=time.monotonic()+3")
rep("    def _sample(self):\n        sample=", '''    def configure_control_quota(self,control,policy):
        require(policy==CONTROL_POLICY,'Exact non-journal quota, no budget substitution')
        self.control=Path(control).resolve()
        require(self.control.is_relative_to(ROOT/'runs'),'Exact new owned sibling control root')
        self.control_guard()
    def control_guard(self,prospective_path=None,prospective_bytes=0):
        if self.control is None:return 0
        total=0;files=0
        for base in (self.control,self.folder):
            for root,dirs,names in os.walk(base):
                if base==self.folder and Path(root)==self.folder:
                    require(all((self.folder/name) in self.worker_slots for name in dirs),'Only registered worker directories')
                    dirs[:]=[]
                for name in dirs:
                    path=Path(root)/name;require(not path.is_symlink(),'No fixed control alias');total+=524288
                for name in names:
                    path=Path(root)/name;require(path.is_file() and not path.is_symlink(),'Exact owned fixed control file')
                    size=path.stat().st_size
                    require(size<=(268435456 if name=='transient_retirement.jsonl' else 16777216),
                            'Bounded fixed control receipt/compact append log')
                    total+=allocated(size);files+=1
                    require(files<=2048,'Bounded fixed-control file table; no per-window leftovers')
        if prospective_path is not None:
            path=Path(prospective_path).resolve()
            require(path.is_relative_to(self.control) or path.parent==self.folder,'Prospective fixed-control output scope')
            old=allocated(path.stat().st_size) if path.exists() else 0
            total+=max(allocated(prospective_bytes)-old,0)+1048576
        require(total<=1073741824,'Fixed one-GiB allocation includes current/prospective temporary files')
        require(total+len(self.worker_slots)*67108864<=4294967296,'Nonjournal four-GiB total plus full reserved worker domain')
        return total
    def reserve_worker_control(self,folder):
        folder=Path(folder).resolve()
        require(self.control is not None and folder.parent==self.folder and not folder.exists()
                and folder not in self.worker_slots and len(self.worker_slots)<48,'Bounded fresh worker-control reservation')
        self.worker_slots.append(folder);self.control_guard()
        write_json(self.control/'GPU_worker_control_slots.json',{'policy':CONTROL_POLICY,
            'slots':[str(path) for path in self.worker_slots],'total_reserved_bytes':len(self.worker_slots)*67108864})
    def _sample(self):
        self.control_guard()
        sample=''')
rep("        self.folder = folder; self.parent_guard = parent_guard; self.config = read_json(config_path)",
"        self.folder = folder; self.parent_guard = parent_guard; self.config = read_json(config_path)\n        require(self.config['control_log_policy']==CONTROL_POLICY,'Same source-bound reserved control quota')")
rep("        folder.mkdir(exist_ok=False); self.log = (folder/'worker.log').open('xb')", 
"        parent_guard.reserve_worker_control(folder)\n        folder.mkdir(exist_ok=False); self.log = (folder/'worker.log').open('xb');self.log_error=None")
rep("stdout=self.log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)",
"stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)\n        self.stdout_bytes=0\n        def bounded_stdout():\n            try:\n                while True:\n                    block=self.child.stdout.read(16384)\n                    if not block:break\n                    require(self.stdout_bytes+len(block)<=CONTROL_POLICY['stdout_log_cap_bytes'],\n                            'Prospective stdout chunk quota; no dropping or truncation')\n                    self.log.write(block);self.log.flush();self.stdout_bytes+=len(block)\n            except BaseException:\n                self.log_error=traceback.format_exc()\n                write_json(folder/'stop.json',{'reason':'owned_stdout_control_quota_failed','error':self.log_error})\n            finally:self.child.stdout.close()\n        self.stdout_thread=threading.Thread(target=bounded_stdout,daemon=True);self.stdout_thread.start()")
rep("        own=self.parent_guard.check()\n        try:",
"        own=self.parent_guard.check();worker_control_guard(self.folder)\n        require(self.log_error is None,'Bounded stdout sink failed; no silent dropped output')\n        try:")
rep("        path = self.folder/('resident_boundary_'+str(self.sequence)+'.json'); write_json(path, value)",
"        window=read_json(response['window_completion']['path'])\n        path=Path(response['window_completion']['path']).parent/'resident_boundary.json'\n        require(path.parent.is_relative_to(Path(self.config['checkpoint_storage_namespace']).resolve()/'transient'),\n                'Per-window sealed boundary only in exact registered current AUX/native')\n        write_json(path,value)")
rep("        self.done.set(); self.monitor.join(timeout=2); self.log.close()",
"        self.done.set(); self.monitor.join(timeout=2);self.stdout_thread.join(timeout=2); self.log.close()\n        require(not self.stdout_thread.is_alive() and self.log_error is None,'Actual stdout EOF and clean joined bounded sink')\n        worker_control_guard(self.folder)")
rep("        self.log.close();self._release_lease()", "        self.stdout_thread.join(timeout=2);self.log.close();self._release_lease()")
rep('from asr18_full_run_parent_controller_v2 import run','from asr18_full_run_parent_controller_v3 import run')
ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
data=new.read_bytes();print({'path':str(new),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'AST_only':True})
