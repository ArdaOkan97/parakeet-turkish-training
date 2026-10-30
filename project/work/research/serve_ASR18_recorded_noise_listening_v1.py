"""One guarded loopback-only static listening page server; no model/audio jobs."""
import http.server
import os
from pathlib import Path
import sys
from ASR18_continuous_data_guard_v5 import DataIOGuard,ROOT
from cpu_preparation_guard_v2 import now
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from atomic_io_retry_v1 import write_json

D=ROOT/'analysis/FastMSS_real_recorded_noise_listening_v1'

def main():
    guard=DataIOGuard('recorded_noise_listening_HTTP',heavy=False,maximum_bytes=512_000_000);guard.wait()
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(D),**kwargs)
        def log_message(self,fmt,*args):
            with (D/'server.log').open('a',encoding='utf-8') as f:f.write(now()+' '+(fmt%args).replace('\n',' ')+'\n')
        def do_GET(self):
            if self.path=='/':self.path='/recorded-noise-turkish-listening.html'
            guard.wait();super().do_GET()
        def copyfile(self,source,output):
            while True:
                guard.wait();block=source.read(65536)
                if not block:break
                guard.pace(len(block));output.write(block)
    class Server(http.server.HTTPServer):
        def service_actions(self):guard.wait()
    chosen=None
    for port in range(8890,8900):
        try:server=Server(('127.0.0.1',port),Handler);chosen=port;break
        except OSError:continue
    assert chosen is not None
    value=dict(kind='FastMSS_recorded_noise_loopback_listening_server_v1',owner='/root/cpu_data_preparation',at_utc=now(),PID=os.getpid(),
        status='running',url=f'http://127.0.0.1:{chosen}/recorded-noise-turkish-listening.html',port=chosen,
        directory=str(D),loopback_only=True,CPU_only=True,production_enabled=False)
    write_json(D/'server_status_v1.json',value);print(value,flush=True)
    server.serve_forever(poll_interval=.25)

if __name__=='__main__':main()
