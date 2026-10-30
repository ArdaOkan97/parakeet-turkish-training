"""Finite final-declaration metadata rejection controls; no final receipt claim."""
import argparse,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from bpe_final_declared_geometry_v2 import check4,actual_exit
from bpe_broad_final_population_v8 import final_population,REQUIRED
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');B=A.parent/'BPE_weak_then_clean_v1'
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only fixture');f.mkdir(exist_ok=False)
    checks=[]
    def reject(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name);return
        raise ValueError('Expected rejection: '+name)
    companion_path=B/'declared_geometry_envelope_concrete_interface_companion_v1.json'
    pin={**record(companion_path),'mtime_ns':companion_path.stat().st_mtime_ns};companion=check4(pin,metadata=True)
    contract=check4(companion['receipt_contract'],metadata=True);producer=check4(companion['producer_source'])
    require(len(contract['required_binding_paths'])==16 and companion['schema_version_value']==1,'Real pinned interface accepted');checks.append('Real contractv2/concrete interface/source four-field bindings validate')
    for key,value in (('bytes',True),('mtime_ns',False),('sha256','0'*64)):
        reject('Reject corrupted/boolean four-field '+key,lambda key=key,value=value:check4({**pin,key:value}))
    reject('Reject omitted signature field',lambda:check4({k:v for k,v in pin.items() if k!='mtime_ns'}))
    reject('Reject extra signature field',lambda:check4({**pin,'complete':True}))
    proof={'kind':'ASR18_source_bound_actual_tool_exit_v1','owner':'/root/cpu_data_preparation','actual_exit_code':0,'actual_tool_chunk_id':'synthetic_protocol_only',
        'published_marker':pin,'native_PID':123,'owned_session':456}
    actual_exit(proof,marker=pin);checks.append('Synthetic actual-exit protocol positive; no invented real producer exit claimed')
    for key,value in (('actual_exit_code',False),('actual_exit_code',1),('native_PID',True),('owned_session',True),('owned_session',None),('actual_tool_chunk_id','')):
        reject('Reject actual-exit '+key+'='+str(value),lambda key=key,value=value:actual_exit({**proof,key:value},marker=pin))
    reject('Wrong published marker rejected',lambda:actual_exit(proof,marker=companion['receipt_contract']))
    reject('Empty final population cannot arm',lambda:final_population({'kind':'ASR18_final_production_population_v8','bindings':{},'complete':True,'pass_count':2}))
    require('declared_geometry_index' in REQUIRED and 'declared_geometry_actual_closed' in REQUIRED and 'declared_geometry_companion' in REQUIRED,'New full declaration requirements');checks.append('Final-population required set includes actual source-bound envelope exit and scalar index')
    text=(S/'bpe_broad_final_population_v8.py').read_text()
    require("geometry['ROOT_native_envelope_review']" in text and "native['ROOT_review']" not in text,'Acyclic review proof chain');checks.append('ROOT review is external to native proof; hash dependency is acyclic')
    require(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow')),'Pure metadata validation');checks.append('No audio/numerical/model imports or native updates')
    write_json(f/'report.json',{'passed':True,'kind':'ASR18_final_declared_metadata_control_v1','checks':checks,'checks_count':len(checks),
        'source':record(__file__),'validator':record(S/'bpe_final_declared_geometry_v2.py'),'population_validator':record(S/'bpe_broad_final_population_v8.py'),
        'real_companion':pin,'full_envelope_or_actual_final_producer_exit_claimed':False,'full729_sidecar_positive_path_not_yet_exercised':True,
        'synthetic_protocol_actual_exit_is_not_a_real_exit_receipt':True,'production_enabled':False,'actual_native_updates':0})
    print(json.dumps({'passed':True,'checks':len(checks),'production_enabled':False}))
if __name__=='__main__':main()
