"""Synthetic stdlib-only arithmetic/context negatives; never actual GPU proof."""
import copy,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from asr18_exact_remaining_family_v1 import remaining,validate_context
from atomic_io_retry_v1 import write_json
checks=[]
def check(name,fn):
    assert fn();checks.append(name)
def reject(name,fn):
    try:fn()
    except (ValueError,TypeError,KeyError,AssertionError):checks.append(name);return
    raise AssertionError(name)
def main():
    observer={'pid':1,'ppid':99,'create_time':10.};launcher={'pid':2,'ppid':1,'create_time':11.};native={'pid':3,'ppid':2,'create_time':12.}
    source={'path':'synthetic-source','bytes':1,'sha256':'a'*64};config={'path':'synthetic-config','bytes':1,'sha256':'b'*64};nonce='c'*32
    c={'kind':'ASR18_actual_owned_GPU_family_context_v1','source':source,'configuration':config,'nonce':nonce,
        'observer':observer,'launcher':launcher,'native':native,'all_live_shims':[launcher],'observed_monotonic':100.}
    def validate(v):return validate_context(v,source=source,configuration=config,nonce=nonce,native=native,now=101.,live=lambda i:i in [observer,launcher,native],descendants=lambda i:[launcher,native])
    check('exact source nonce lineage context',lambda:validate(c)==(observer,[launcher]))
    check('remaining future allowance avoids counted current family',lambda:remaining(20_000_000,[8_000_000,4_000_000])==160_000_000)
    check('same projected full future family arithmetic',lambda:32_000_000+remaining(20_000_000,[8_000_000,4_000_000])==192_000_000)
    check('all hard caps retained at fully occupied family',lambda:remaining(128_000_000,[64_000_000])==0)
    for name,path,value in [('stale',('observed_monotonic',),98.),('future',('observed_monotonic',),102.),('wrong_nonce',('nonce',),'d'*32),
        ('wrong_source',('source','sha256'),'d'*64),('wrong_config',('configuration','sha256'),'d'*64),('wrong_native_PID',('native','pid'),4),
        ('wrong_native_ctime',('native','create_time'),13.),('boolean_PID',('native','pid'),True),('missing_ALL_shims',('all_live_shims',),[]),
        ('native_counted_as_shim',('all_live_shims',),[launcher,native]),('duplicate_shim',('all_live_shims',),[launcher,launcher])]:
        v=copy.deepcopy(c);part=v
        for key in path[:-1]:part=part[key]
        part[path[-1]]=value;reject(name,lambda v=v:validate(v))
    reject('parent cap unchanged',lambda:remaining(128_000_001,[1]));reject('all-shim cap unchanged',lambda:remaining(1,[64_000_000,1]))
    reject('unavailable private',lambda:remaining(0,[1]));reject('bool private rejected',lambda:remaining(True,[1]))
    reject('live identity unavailable',lambda:validate_context(c,source=source,configuration=config,nonce=nonce,native=native,now=101.,live=lambda i:False,descendants=lambda i:[launcher,native]))
    reject('uncaptured new descendant',lambda:validate_context(c,source=source,configuration=config,nonce=nonce,native=native,now=101.,live=lambda i:True,descendants=lambda i:[launcher,native,{'pid':4,'ppid':2,'create_time':13.}]))
    folder=Path(sys.argv[1]);folder.mkdir(exist_ok=False)
    write_json(folder/'report.json',{'kind':'ASR18_SYNTHETIC_remaining_family_arithmetic_validation_v1','passed':True,'checks':checks,
        'actual_GPU_or_ML_used':False,'fake_actual_process_or_GPU_evidence_written':False,'unchanged_caps':[128000000,64000000]})
    print({'checks':len(checks),'passed':True})
if __name__=='__main__':main()
