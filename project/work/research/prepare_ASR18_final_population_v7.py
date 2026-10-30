"""Acyclic final-review binding successor; v6 remains unadmitted history."""
import sys,json
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    old=S/'bpe_broad_final_population_v6.py';target=S/'bpe_broad_final_population_v7.py';require(not target.exists(),'Fresh unadmitted successor')
    source=old.read_text();before="reviewed=load_bound(native['ROOT_review'],guard=guard)";after="reviewed=load_bound(geometry['ROOT_native_envelope_review'],guard=guard)"
    require(source.count(before)==1,'Exact delta required');source=source.replace(before,after).replace('ASR18_final_production_population_v6','ASR18_final_production_population_v7')
    target.write_text(source)
    write_json(A/'final_population_declared_successor_preparation_v7.json',{'source':record(target),'prior_unadmitted_source':record(old),
        'v6_issue':'Native admission referring to ROOT review which itself binds admission creates an impossible cryptographic cycle',
        'v7_rule':'Outer geometry admission binds both native proof and ROOT review; review binds native proof, native proof never binds its future review',
        'production_enabled':False,'full_final_source_review_pending':True,'final_population_or_geometry_claimed':False})
    print(json.dumps({'inactive_successor':record(target),'production_enabled':False}))
if __name__=='__main__':main()
