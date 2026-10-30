"""Source-only AUX lifetime routing; no target import or ML/job."""
from pathlib import Path
import ast,hashlib
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'
def emit(source,target,changes):
    assert not target.exists();text=source.read_text(encoding='utf-8-sig')
    for before,after in changes:
        assert text.count(before)==1,before[:90];text=text.replace(before,after)
    ast.parse(text);compile(text,str(target),'exec');target.write_text(text,encoding='utf-8',newline='\n')
    data=target.read_bytes();print({'path':str(target),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'AST_only':True})
emit(S/'asr18_full_run_native_owner_v3.py',S/'asr18_full_run_native_owner_v4.py',[
 ("write_json(self.folder/'native_progress.json'","write_json(self.active_folder/'native_progress.json'"),
])
emit(R/'asr18_full_run_owned_GPU_runtime_v3.py',R/'asr18_full_run_owned_GPU_runtime_v4.py',[
 ("    def emit(name,**extra):\n        value=stage(folder,name,torch,**extra)",
  "    stage_folder=folder\n    def emit(name,**extra):\n        value=stage(stage_folder,name,torch,**extra)"),
 ("from asr18_full_run_native_owner_v3 import FullRunNativeOwner","from asr18_full_run_native_owner_v4 import FullRunNativeOwner"),
 ("                write_json(folder/'response.json',{'passed':True,'sequence':sequence,'nonce':command['nonce'],'actual_process_exit_pending':True});break",
  "                stage_folder=folder\n                write_json(folder/'response.json',{'passed':True,'sequence':sequence,'nonce':command['nonce'],'actual_process_exit_pending':True});break"),
 ("            if command['kind']=='run_prepared_window':\n                complete=engine.run_window(command['window'])",
  "            if command['kind']=='run_prepared_window':\n                stage_folder=Path(command['window']['auxiliary_workspace']['workspace'])/'native'\n                complete=engine.run_window(command['window'])"),
 ("            ip=folder/('idle_ack_'+str(sequence)+'.json');write_json(ip,idle)",
  "            ip=engine.active_folder/'idle_ack.json';write_json(ip,idle)"),
 ("            release_path=folder/('checkpoint_release_ack_'+str(sequence)+'.json');write_json(release_path,released)",
  "            release_path=engine.active_folder/'checkpoint_release_ack.json';write_json(release_path,released)"),
])
