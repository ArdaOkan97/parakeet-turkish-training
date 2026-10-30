"""Source-only exact owned atomic-file observation and global-stage bound."""
from pathlib import Path
import ast,hashlib
P=Path(__file__).resolve().parents[2];R=P/'work/research'
old=R/'run_ASR18_full_run_owner_v5.py';new=R/'run_ASR18_full_run_owner_v6.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
before="""            path=Path(root)/name;require(path.is_file() and not path.is_symlink(),'Only exact regular worker control files')
            require(path.suffix.lower() in ('.json','.jsonl','.log','.txt','.tmp'),
                    'No tensor/audio/model/unknown files in control-log domain')
            size=path.stat().st_size;limit=(8388608 if name in ('stages.jsonl','worker.log') else 1048576)"""
after="""            path=Path(root)/name
            try:info=path.lstat()
            except FileNotFoundError:
                require('.atomic-v1-' in name and name.endswith('.tmp'),'Only exact atomic temporary may disappear during observation');continue
            require(__import__('stat').S_ISREG(info.st_mode) and not path.is_symlink(),'Only exact regular worker control files')
            require(path.suffix.lower() in ('.json','.jsonl','.log','.txt','.tmp'),
                    'No tensor/audio/model/unknown files in control-log domain')
            size=info.st_size;limit=(8388608 if name in ('stages.jsonl','worker.log') else 1048576)"""
assert text.count(before)==1;text=text.replace(before,after)
before="""                    path=Path(root)/name;require(path.is_file() and not path.is_symlink(),'Exact owned fixed control file')
                    size=path.stat().st_size"""
after="""                    path=Path(root)/name
                    try:info=path.lstat()
                    except FileNotFoundError:
                        require(('.atomic-v1-' in name and name.endswith('.tmp')) or '.proof.' in name or name.endswith('.plan.json'),
                                'Only source-owned atomic/returned-operation temporaries may disappear');continue
                    require(__import__('stat').S_ISREG(info.st_mode) and not path.is_symlink(),'Exact owned fixed control file')
                    size=info.st_size"""
assert text.count(before)==1;text=text.replace(before,after)
before="""            log = getattr(candidate, 'log', None)
            if log is not None: log.close()"""
after="""            sink=getattr(candidate,'stdout_thread',None)
            if sink is not None:sink.join(timeout=2)
            log = getattr(candidate, 'log', None)
            if log is not None: log.close()"""
assert text.count(before)==1;text=text.replace(before,after)
ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
data=new.read_bytes();print({'path':str(new),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'AST_only':True})
old=R/'asr18_full_run_owned_GPU_runtime_v4.py';new=R/'asr18_full_run_owned_GPU_runtime_v5.py'
assert not new.exists();text=old.read_text(encoding='utf-8-sig')
text=text.replace('import argparse,copy,gc,os,sys,time','import argparse,copy,gc,os,sys,time,json')
before="""    def emit(name,**extra):
        value=stage(stage_folder,name,torch,**extra)"""
after="""    def emit(name,**extra):
        if stage_folder==folder:
            from run_ASR18_full_run_owner_v6 import worker_control_guard
            # Original fixed telemetry plus <=64KiB explicit extras fits the
            # reserved1MiB atomic/global-stage write allowance. No GPU saving.
            require(len(json.dumps(extra,ensure_ascii=False).encode('utf-8'))<=65536,'Bounded global stage extras')
            path=folder/'stages.jsonl'
            require((path.stat().st_size if path.exists() else 0)+1048576<=8388608,
                    'Prospective global startup/release stage-log bound')
            worker_control_guard(folder)
        value=stage(stage_folder,name,torch,**extra)
        if stage_folder==folder:worker_control_guard(folder)"""
assert text.count(before)==1;text=text.replace(before,after)
ast.parse(text);compile(text,str(new),'exec');new.write_text(text,encoding='utf-8',newline='\n')
data=new.read_bytes();print({'path':str(new),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'AST_only':True})
