import argparse,json,time,psutil,os
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--worker',action='store_true')
parser.add_argument('--configuration',required=True)
parser.add_argument('--private-limit',required=True)
args=parser.parse_args()
assert args.worker and args.private_limit=='8000000000'
config=json.loads(Path(args.configuration).read_text())
p=psutil.Process()
ready=Path(config['ready']);tmp=Path(str(ready)+'.owned.tmp');f=tmp.open('x');json.dump({'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()},f);f.flush();os.fsync(f.fileno());f.close();os.rename(tmp,ready)
start=time.monotonic()
while not Path(config['done']).exists() and time.monotonic()-start<60:time.sleep(.02)
