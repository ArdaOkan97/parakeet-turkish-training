"""Small CPU guards for fixed-set batching/alignment and paired memory policy."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'outputs/parakeet_turkish'))
import torch
import profile_asr_inference_v1 as worker
torch.set_num_threads(2)
checks=0
def check(ok):
    global checks
    if not ok:raise AssertionError('CPUguard '+str(checks+1))
    checks+=1
rows=worker.selected_rows(worker.ROOT/'work/data/validation.jsonl')
check(len(rows)==16 and sum(row['language']=='tr' for row in rows)==8)
for size in (1,4,8,16):
    normal=worker.groups(list(range(16)),size)
    reverse=worker.groups(list(reversed(range(16))),size)
    check(sum(normal,[])==list(range(16)))
    check(sum(reverse,[])==list(reversed(range(16))))
    values=[None]*16
    for group in reverse:
        for index in group:values[index]='text '+str(index)
    check(worker.transcript_comparison(['text '+str(i) for i in range(16)],values,rows)['exact_transcripts_equal'])
text=['text '+str(i) for i in range(16)]
changed=text.copy();changed[7]='different'
check(worker.transcript_comparison(text,changed,rows)['differences'][0]['source_id']==rows[7]['source_id'])
check(worker.transcript_comparison(text,changed,rows)['changed_utterances']==1)
for fraction,headroom,expected in ((.949,3e9,False),(.95,3e9,True),(.1,1.999e9,True),(.94,2e9,False)):
    check(worker.critical_commit({'system_memory':{'available':True,'commit_fraction':fraction,'commit_available_bytes':headroom}})==expected)
check(worker.critical_commit({'calling_process_commit_memory':{'available':True,'ullAvailPageFile_bytes':1.99e9}}))
check(not worker.critical_commit({'available':False}))
rejected=0
for order,size in (([0]*16,8),(list(range(15)),4),(list(range(16)),2)):
    try:worker.groups(order,size)
    except RuntimeError:rejected+=1
    else:raise AssertionError('Corrupt group accepted')
check(rejected==3)
check(worker.sha(worker.capture.__file__)==worker.EXPECTED_LOADER_SHA)
check(not torch.cuda.is_initialized())
output=worker.OUTPUT/'asr_inference_cpu_guards_v1.json'
worker.save(output,{'schema':1,'passed':True,'checks':checks,'rejected_corruptions':rejected,
    'producer_sha256':worker.sha(__file__),'worker_sha256':worker.sha(worker.__file__),
    'actual_preparation_sha256':worker.sha(worker.OUTPUT/'asr_inference_preparation_v1.json'),
    'cuda_initialized':torch.cuda.is_initialized(),'scope':'CPU fixed16-source grouping/identity, permutation restoration, retained transcript changes and paired commitment boundary guards; no actual inference'})
print(json.dumps({'passed':True,'checks':checks,'rejected':rejected}))
