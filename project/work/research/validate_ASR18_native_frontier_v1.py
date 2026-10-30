import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from asr18_cleaner_native_frontier_evidence_v1 import publish
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only frontier fixture');f.mkdir(exist_ok=False);checks=[]
    def check(x,n):require(x,n);checks.append(n)
    opt=SimpleNamespace(updates=20);v=publish(f,opt,error='intentional before Adam')
    check(v['completed_updates']==20 and v['native_updates_this_child']==0 and not v['actual_positive_Adam_observed_in_owned_optimizer'],'Restored source20 is not a new optimizer update')
    opt.updates=21;v=publish(f,opt,error='intentional native post-Adam failure')
    check(v['native_updates_this_child']==1 and not v['native_return_complete'] and not v['journal_sampler_committed'],'Actual positive21 remains visible even without returned native receipt')
    v=publish(f,opt,native={'FIXTURE_NOT_NATIVE':True})
    check(v['native_return_complete'] and v['native_updates_this_child']==1,'Positive native return published before monitor closure')
    try:raise RuntimeError('intentional monitor finish failure')
    except RuntimeError:check(json.loads((f/'native_progress.json').read_text())['native_updates_this_child']==1,'Monitor closure error cannot hide already-recorded positive Adam')
    v=publish(f,opt,native={'FIXTURE_NOT_NATIVE':True},committed=True)
    check(v['journal_sampler_committed'],'Sampler journal flag follows explicit actual commit only')
    try:publish(f,SimpleNamespace(updates=20),committed=True)
    except ValueError:checks.append('No journal commit permitted for only-restored source20')
    else:raise ValueError('False commit accepted')
    check(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow')),'No model/numerical import')
    write_json(f/'report.json',{'passed':True,'checks':checks,'source':record(__file__),'helper':record(S/'asr18_cleaner_native_frontier_evidence_v1.py'),
        'actual_native_updates':0,'synthetic_optimizer_and_injected_monitor_failure_only':True,'production_ready':False})
    print(json.dumps({'passed':True,'checks':len(checks),'actual_native_updates':0}))
if __name__=='__main__':main()
