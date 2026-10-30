import ast,hashlib,json,os
from pathlib import Path
F=Path(__file__).parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
specs=[('asr18_fresh_single_open_metadata_signature_v1.py','asr18_fresh_single_open_metadata_signature_v2.py',[
 ('import ctypes,os\n','import ctypes,os,threading\n'),
 ("        self.api=ctypes.WinDLL('kernel32',use_last_error=True)\n","        self.local=threading.local() # At most one bounded64KiB buffer per participating thread.\n        self.api=ctypes.WinDLL('kernel32',use_last_error=True)\n"),
 ("            needed=self.api.GetFinalPathNameByHandleW(handle,None,0,0)\n            if not needed:raise ctypes.WinError(ctypes.get_last_error())\n            if needed>=MAX_FINAL_PATH_CHARS:raise UnsupportedFastSignature('Bounded normal final pathname')\n            buffer=ctypes.create_unicode_buffer(needed+1)\n",
  "            buffer=getattr(self.local,'final_path_buffer',None)\n            if buffer is None:\n                buffer=ctypes.create_unicode_buffer(MAX_FINAL_PATH_CHARS)\n                self.local.final_path_buffer=buffer\n")]),
 ('asr18_fresh_metadata_signature_scope_v1.py','asr18_fresh_metadata_signature_scope_v2.py',[
 ('import threading\n','import threading\nfrom pathlib import Path\nfrom bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks\n'),
 ('from asr18_fresh_single_open_metadata_signature_v1 import FreshSingleOpenSignature,ORIGINAL_SIGNATURE\n','from asr18_fresh_single_open_metadata_signature_v2 import FreshSingleOpenSignature,ORIGINAL_SIGNATURE\n'),
 ("        resolver=context.resolver\n","        if not callable(private_worker_closed):raise ValueError('Callable original private-worker closure check')\n        resolver=context.resolver\n"),
 ("        if len(resolver.metadata_locks.handles)!=3:raise ValueError('All three original denywrite handles remain live')\n        expected=[*resolver.signatures,resolver.geometry.signature]\n",
  "        expected=[*resolver.signatures,resolver.geometry.signature]\n        if (type(resolver.metadata_locks)is not FinalMetadataReadLocks\n            or len(resolver.metadata_locks.handles)!=3\n            or resolver.metadata_locks.paths!=[Path(v['path']) for v in expected]):\n            raise ValueError('All three exact original denywrite lock paths and handles remain live')\n"),
 ("        if not self.private_worker_closed():\n",
  "        try:joined=self.private_worker_closed()\n        except BaseException:\n            self.deferred_to_abnormal_exit=True\n            if exc_type is None:raise\n            return False\n        if joined is not True:\n")]),
 ('asr18_observe_original_gc_collect_v1.py','asr18_observe_original_gc_collect_v2.py',[
 ('import gc,threading,time\n','import gc,threading,time,types\n'),
 ("        if gc.collect is not ORIGINAL_COLLECT:raise ValueError('Exact unchanged original builtin before observation')\n",
  "        if (not isinstance(ORIGINAL_COLLECT,types.BuiltinFunctionType)\n            or ORIGINAL_COLLECT.__name__!='collect' or ORIGINAL_COLLECT.__module__!='gc'):\n            raise ValueError('Captured original must be the real gc.collect builtin')\n        if gc.collect is not ORIGINAL_COLLECT:raise ValueError('Exact unchanged original builtin before observation')\n")])]
ledger={'kind':'ASR18_inactive_signature_scope_collect_observer_v2_exact_edit_ledger_v1','newline_normalization':'NONE; UTF8 LF exact','records':[],'v1_unexecuted_preserved':True,'production_change':False}
for oldname,newname,edits in specs:
    old=F/oldname;new=F/newname;raw=old.read_bytes();text=raw.decode('utf-8')
    for a,b in edits:
        assert text.count(a)==1;text=text.replace(a,b,1)
    # Keep path-class import explicit in the frozen successor, declared in same edit.
    ast.parse(text);assert not new.exists();new.write_bytes(text.encode('utf-8'))
    inverse=text
    for a,b in reversed(edits):assert inverse.count(b)==1;inverse=inverse.replace(b,a,1)
    assert inverse.encode('utf-8')==raw
    ledger['records'].append({'old':pin(old),'new':pin(new),'edits':[{'old_literal':a,'new_literal':b,'expected_count':1}for a,b in edits],'full_raw_inverse_exact':True})
out=F/'inactive_signature_scope_collect_observer_v2_exact_ledger.json'
with out.open('x',encoding='utf-8',newline='\n')as f:json.dump(ledger,f,indent=2);f.write('\n')
print(json.dumps({'ledger':pin(out),'successors':[r['new']for r in ledger['records']],'actual_control_execution':False}))
