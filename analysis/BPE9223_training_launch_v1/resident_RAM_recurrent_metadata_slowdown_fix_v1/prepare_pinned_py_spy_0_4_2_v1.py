import datetime, hashlib, json, os, subprocess, urllib.request, zipfile
from pathlib import Path

F=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1\resident_RAM_recurrent_metadata_slowdown_fix_v1')
OUT=F/'py_spy_0_4_2_win_amd64_v1'
URL='https://files.pythonhosted.org/packages/6f/ed/1409cdb557e558a6c98003ab12fdd4284699e158c167c187cb0f124eea4c/py_spy-0.4.2-py2.py3-none-win_amd64.whl'
SHA='8b06a353c177677e4e1701b288d8c58e2f8d4208ee81a8048d9f72ba800918f8'

def pin(p):
    b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,v):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')

assert os.environ['TORCH_EXTENSIONS_DIR'].lower().startswith('e:')
assert not OUT.exists(), 'Immutable fresh preparation output required'
OUT.mkdir()
receipt={'kind':'ASR18_pinned_py_spy_standalone_preparation_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'official_release':'https://pypi.org/project/py-spy/0.4.2/','url':URL,'expected_wheel_sha256':SHA,'installation_performed':False,'target_process_attachment_performed':False,'toy_or_live_run_performed':False,'passed':False,'error':None}
try:
    wheel=OUT/'py_spy-0.4.2-py2.py3-none-win_amd64.whl'
    with urllib.request.urlopen(URL,timeout=30) as response,wheel.open('xb') as f:
        total=0
        while True:
            b=response.read(65536)
            if not b:break
            total+=len(b);assert total<=10*1024*1024,'Wheel byte bound exceeded';f.write(b)
    receipt['wheel']=pin(wheel);assert receipt['wheel']['sha256']==SHA
    with zipfile.ZipFile(wheel) as z:
        names=z.namelist();receipt['wheel_members']=names
        exe_names=[n for n in names if n.endswith('/py-spy.exe') or n=='py-spy.exe']
        assert len(exe_names)==1,exe_names
        info=z.getinfo(exe_names[0]);assert 0<info.file_size<=20*1024*1024
        executable=OUT/'py-spy.exe'
        with executable.open('xb') as f:f.write(z.read(info))
        receipt['extracted_member']=exe_names[0]
    receipt['executable']=pin(executable)
    receipt['preparation_commands']={}
    for name,args in [('version',['--version']),('record_help',['record','--help'])]:
        with (OUT/(name+'.stdout.txt')).open('xb') as o,(OUT/(name+'.stderr.txt')).open('xb') as e:
            r=subprocess.run([str(executable)]+args,cwd=OUT,stdout=o,stderr=e,timeout=15,check=False,creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        receipt['preparation_commands'][name]={'arguments':args,'returncode':r.returncode,'stdout':pin(OUT/(name+'.stdout.txt')),'stderr':pin(OUT/(name+'.stderr.txt'))}
        assert r.returncode==0,(name,r.returncode)
    receipt['storage_configuration']=pin(Path(r'E:\gpu_training_scratchpad\storage_config.json'))
    receipt['storage_script']=pin(Path(r'E:\gpu_training_scratchpad\use_storage.ps1'))
    receipt['preparation_source']=pin(Path(__file__))
    receipt['passed']=True
except BaseException as e:
    receipt['error']={'type':type(e).__name__,'message':str(e)}
finally:
    write(OUT/'preparation_receipt.json',receipt)
print(json.dumps({'passed':receipt['passed'],'receipt':pin(OUT/'preparation_receipt.json'),'wheel':receipt.get('wheel'),'executable':receipt.get('executable'),'error':receipt['error']}))
raise SystemExit(0 if receipt['passed'] else 1)
