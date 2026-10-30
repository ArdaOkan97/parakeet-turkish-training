import argparse,json,time,psutil
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--worker',action='store_true')
parser.add_argument('--configuration',required=True)
parser.add_argument('--private-limit',required=True)
args=parser.parse_args()
assert args.worker and args.private_limit=='8000000000'
config=json.loads(Path(args.configuration).read_text())
p=psutil.Process()
Path(config['ready']).write_text(json.dumps({'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}))
start=time.monotonic()
while not Path(config['done']).exists() and time.monotonic()-start<60:time.sleep(.02)
