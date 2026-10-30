"""Finite final-denominator rejection controls; no live catalog writes."""
import argparse,json,sqlite3,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from bpe_final_denominator_v1 import audit
from bpe_broad_final_population_v4 import final_population
from tdt_full_native_allocation_budget_v1 import allocation_budget

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);args=parser.parse_args();folder=Path(args.folder)
    require(folder.is_relative_to(Path(r'E:\gpu_training_scratchpad')),'E-only fixture');folder.mkdir(exist_ok=False)
    clips=folder/'catalog.sqlite';labels=folder/'labels.sqlite'
    with sqlite3.connect(clips) as c:
        c.execute('CREATE TABLE clips(ordinal INTEGER PRIMARY KEY)');c.executemany('INSERT INTO clips VALUES(?)',[(v,) for v in (1,2,3,4)])
    with sqlite3.connect(labels) as c:
        c.execute('CREATE TABLE labels(ordinal INTEGER PRIMARY KEY,target_tokens INTEGER,target_ids_u32 BLOB)')
        c.executemany('INSERT INTO labels VALUES(?,?,?)',[(v,v,bytes(4*v)) for v in (1,2,3,4)])
    results={'positive_contiguous_denominator':audit(clips,labels,rows=4,tokens=10)['passed']}
    def rejected(name,call):
        try:call()
        except (ValueError,KeyError):results[name]=True
        else:raise ValueError('Expected fail-closed rejection: '+name)
    rejected('wrong_final_token_sum',lambda:audit(clips,labels,rows=4,tokens=11))
    with sqlite3.connect(clips) as c:c.execute('UPDATE clips SET ordinal=5 WHERE ordinal=2')
    rejected('hole_and_extra_ordinal',lambda:audit(clips,labels,rows=4,tokens=10))
    with sqlite3.connect(clips) as c:c.execute('UPDATE clips SET ordinal=2 WHERE ordinal=5')
    with sqlite3.connect(labels) as c:c.execute("UPDATE labels SET target_ids_u32=x'00' WHERE ordinal=3")
    rejected('malformed_full_target_BLOB',lambda:audit(clips,labels,rows=4,tokens=10))
    rejected('empty_final_bindings',lambda:final_population({'kind':'ASR18_final_production_population_v4','complete':True,'pass_count':2,'bindings':{}}))
    rejected('missing_required_final_bindings',lambda:final_population({'kind':'ASR18_final_production_population_v4','complete':True,'pass_count':2,'bindings':{'finalization_contract':{}}}))
    budget=allocation_budget(1991,574,projected_elements=(1991+575)*640,head_elements=8716*641,hidden_width=640)
    results['full_hard_conservative_budget_supported']=budget['total_budget_bytes']<1_700_000_000 and budget['padded_full_cell_bytes']>92
    rejected('oversize_full_DP_total_rejected',lambda:allocation_budget(100000,574,projected_elements=100575*640,head_elements=8716*641,hidden_width=640))
    rejected('unsupported_native_U_rejected',lambda:allocation_budget(10,1024,projected_elements=1035*640,head_elements=8716*641,hidden_width=640))
    sys.path.append(str(P/'work/training-env/Lib/site-packages'))
    from jsonschema import Draft202012Validator,ValidationError
    from bpe_final_carrier_validation_v2 import check
    B=Path(r'E:\gpu_training_scratchpad\analysis\BPE_weak_then_clean_v1')
    companion=json.loads((B/'KIRAAT_FLAC_normalized_stage_contract_v1.json').read_text())
    prior=json.loads((B/'carrier_provenance_normalization_actual_v1/report.json').read_text())
    for item in prior['controls']:
        if 'normalized_stage' not in item:continue
        stage=item['normalized_stage'];proof=check(stage['actual_exit_proof'],guard=lambda:None,metadata=True)
        footer=check(stage['native_guard_closure'],guard=lambda:None,metadata=True)
        Draft202012Validator(companion['stage_proof_schema']).validate(proof)
        Draft202012Validator(companion['native_guard_schema']).validate(footer)
        require(proof['native_PID']==footer['native_PID']==stage['native_PID'] and proof['normalizer_source']==companion['normalizer_source'],'Real actual-provenance identity differs')
        results['real_normalized_'+item['name']]=True
    for name,key,value in (('reject_actual_exit1','actual_exit_code',1),('reject_missing_watchdog','watchdog_clean',False)):
        changed={**footer,key:value}
        try:Draft202012Validator(companion['native_guard_schema']).validate(changed)
        except ValidationError:results[name]=True
        else:raise ValueError('Invalid actual guard footer accepted')
    require(all(results.values()),'Every meaningful denominator/budget/provenance rejection required')
    write_json(folder/'report.json',{'passed':True,'checks':results,'checks_count':len(results),'full_hard_allocation_budget':budget,
        'source':record(__file__),'final_population_validator':record(P/'outputs/parakeet_turkish/bpe_broad_final_population_v3.py'),
        'live_catalog_or_labels_modified':False,'full_population_or_geometry_claimed':False,'actual_optimizer_updates':0})
    print(json.dumps({'passed':True,'checks':len(results)}))

if __name__=='__main__':main()
