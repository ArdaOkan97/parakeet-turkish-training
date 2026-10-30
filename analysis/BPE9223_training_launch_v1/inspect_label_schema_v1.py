import importlib,json,sys
from pathlib import Path
L=Path(r'E:\gpu_training_scratchpad\analysis\Turkish_labels_BPE_circumflex_v1')
sys.path.insert(0,str(L));m=importlib.import_module('audit_and_fit_append_only_v5')
N=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1');m.AREA=N
def main():
 g=m.Guard();g.phase='small_readonly_existing_label_and_geometry_schema'
 out={}
 for name,path in [('broad',m.BROAD_LABELS),('cleaner',m.CLEAN_LABELS),('geometry',m.ROOT/'indexes/ASR18_declared_training_geometry_v1.sqlite')]:
  db=m.ro(path);out[name]={}
  for table,sql in db.execute("SELECT name,sql FROM sqlite_master WHERE type='table'"):
   row=db.execute('SELECT * FROM '+table+' LIMIT 1').fetchone()
   out[name][table]={'sql':sql,'columns':[x[1] for x in db.execute('PRAGMA table_info('+table+')')],
     'example':[{'BLOB_bytes':len(x)} if isinstance(x,bytes) else x for x in row] if row else None}
  db.close()
 m.write('existing_schema_readonly.json',out);g.publish(passed=True)
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
