"""CPU fail-closed population, bijection, exact-tail and journal-prefix fixture."""
import argparse,json,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from bpe_broad_production_state_v1 import CompletePassCursor,CommitJournal,permutation,require,final_population
from bpe_broad_production_v1 import production_preflight
from atomic_io_retry_v1 import write_json

def rejects(call):
    try:call()
    except (ValueError,KeyError):return True
    raise AssertionError('Unsafe admission did not fail closed')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();folder=Path(a.output);folder.mkdir(parents=True,exist_ok=False)
    checks=0
    for size in (1,2,3,7,17,128,257,1000):
        for seed in (0,20261017):
            values=[permutation(i,size,seed) for i in range(size)]
            require(sorted(values)==list(range(size)),'Exact non-power-of-two full-pass permutation');checks+=1
        cursor=CompletePassCursor(population=size,identity={'final_SHA':'fixture'},seed=19)
        seen={0:[],1:[]}
        while cursor.remaining:
            prepared=cursor.preview(min(14,cursor.remaining));before=cursor.state_dict()
            # Preparation has no effect, including failed/repeated preparation.
            require(cursor.state_dict()==before and cursor.preview(len(prepared))==prepared,'Prepared draw consumed committed cursor')
            restored=CompletePassCursor(population=size,identity={'final_SHA':'fixture'},seed=19);restored.load_state_dict(before)
            require(restored.preview(len(prepared))==prepared,'Next draw changed after actual state restore')
            for item in prepared:seen[item['epoch']].append(item['ordinal'])
            cursor.commit(prepared);checks+=1
        require(all(sorted(v)==list(range(1,size+1)) for v in seen.values()) and cursor.coverage()['complete'],'Two full passes must include every ordinal once')
        rejects(lambda:cursor.preview(1));checks+=2
    partial={'complete':False,'population_mean_U':53.1}
    rejects(lambda:final_population(partial,verify_record=lambda item:True));checks+=1
    fake={'kind':'ASR18_final_production_population_v1','complete':True,'all_pinned_sources_complete':True,'metadata_native_exit_code':0,'labels_native_exit_code':0,
        'retained_rows':3,'label_rows':3,'total_target_tokens':30,'population_mean_U':10.,'quality_filters':[],'language_filters':[],'pass_count':2,
        'runtime_heldout_PCM_binding_complete':True,'unresolved_geometry_rows':0,'bindings':{'source':{'sha':'frozen'}}}
    require(final_population(fake,verify_record=lambda item:True)==(3,10.),'Final full sum/count interface');checks+=1
    for key,value in (('label_rows',2),('population_mean_U',9.9),('all_pinned_sources_complete',False),('unresolved_geometry_rows',1),('runtime_heldout_PCM_binding_complete',False)):
        rejects(lambda key=key,value=value:final_population({**fake,key:value},verify_record=lambda item:True));checks+=1
    rejects(lambda:final_population(fake,verify_record=lambda item:False));checks+=1
    config=json.loads((PROJECT/'outputs/parakeet_turkish/bpe_broad_production_v1.disabled.json').read_text());rejects(lambda:production_preflight(config));checks+=1
    journal=CommitJournal(folder/'journal.jsonl');empty=journal.binding();one=journal.append({'update':1,'draws':14})
    require(journal.verify(empty) and journal.verify(one),'Valid committed prefix rejected');checks+=1
    journal.append({'prepared_uncommitted_tail_test':True});require(journal.verify(one),'Tail must not consume checkpoint prefix');checks+=1
    altered={**one,'sha256':'0'*64};require(not journal.verify(altered),'Changed journal prefix admitted');checks+=1
    require('torch' not in sys.modules and 'transformers' not in sys.modules,'No model/runtime import')
    write_json(folder/'report.json',{'passed':True,'meaningful_checks':checks,'two_complete_passes_bijective':True,'failed_preparation_advances_zero_draws':True,
        'exact_tail_no_wrap_or_discard':True,'restore_next_draw_exact':True,'partial_normalization_rejected':True,'current_production_preflight_rejected':True,'GPU_used':False})
    print(json.dumps({'passed':True,'checks':checks}),flush=True)

if __name__=='__main__':main()
