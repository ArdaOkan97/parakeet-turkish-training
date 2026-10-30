"""Strict count/kind/exit-type successor; prior sources stay unadmitted."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def change(text,old,new):require(text.count(old)==1,'Exact source delta required');return text.replace(old,new)
def main():
    source=S/'bpe_final_declared_geometry_v1.py';new=S/'bpe_final_declared_geometry_v2.py';require(not new.exists(),'Fresh geometry successor')
    code=source.read_text();code=change(code,"require(type(sidecars) is dict and set(sidecars)==set(contract['KIRAAT_sidecar_inventory_exact_keys']) and sidecars['complete'] is True",
        "require(type(sidecars) is dict and set(sidecars)==set(contract['KIRAAT_sidecar_inventory_exact_keys']) and sidecars['kind']=='ASR18_complete_KIRAAT_declared_geometry_sidecar_inventory_v1' and sidecars['complete'] is True")
    code=change(code,"    rows=envelope['population']['retained_rows'];tokens=envelope['population']['total_target_tokens']",
        "    require(type(envelope['population']) is dict and set(envelope['population'])==set(contract['population_exact_keys'])\n        and all(type(v) is int and v>=0 for v in envelope['population'].values()),'Exact scalar retained envelope integers required; no bool counts')\n    require(type(envelope['groups']) is dict and set(envelope['groups'])==set(contract['group_keys_exact'])\n        and all(type(g) is dict and set(g)==set(contract['group_scalar_keys_exact']) and all(type(v) is int and v>=0 for v in g.values()) for g in envelope['groups'].values()),\n        'Exact per-source scalar group integer fields required')\n    rows=envelope['population']['retained_rows'];tokens=envelope['population']['total_target_tokens']")
    new.write_text(code)
    old=S/'bpe_broad_final_population_v7.py';target=S/'bpe_broad_final_population_v8.py';require(not target.exists(),'Fresh population successor')
    text=old.read_text().replace('bpe_final_declared_geometry_v1','bpe_final_declared_geometry_v2').replace('ASR18_final_production_population_v7','ASR18_final_production_population_v8')
    for who in ('proof','acquisition','headers','carrier_exit'):
        needle=who+"['actual_exit_code']==0";replacement="type("+who+"['actual_exit_code']) is int and "+needle
        text=change(text,needle,replacement)
    target.write_text(text)
    fixture=P/'work/research/validate_ASR18_final_declared_metadata_v1.py';nextfixture=P/'work/research/validate_ASR18_final_declared_metadata_v2.py';require(not nextfixture.exists(),'Fresh metadata fixture')
    text=fixture.read_text().replace('bpe_final_declared_geometry_v1','bpe_final_declared_geometry_v2').replace('bpe_broad_final_population_v7','bpe_broad_final_population_v8').replace('ASR18_final_production_population_v7','ASR18_final_production_population_v8')
    nextfixture.write_text(text)
    write_json(A/'final_declared_strict_metadata_preparation_v8.json',{'source':record(target),'geometry_validator':record(new),'fixture':record(nextfixture),
        'prior_sources_preserved':[record(old),record(source)],'changes':'Exact sidecar-inventory kind, exact integer population/group counts, exact integer actual-exit codes for all final workers',
        'production_enabled':False,'full729_positive_path_exercised':False,'finite_cleaner_review_manifest_unchanged':True})
    print(json.dumps({'future_population_source':record(target),'production_enabled':False}))
if __name__=='__main__':main()
