"""Bind a zero-Adam component profile and the actual CPU frontend fixture."""
import ast,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path[:0]=[str(S),str(P/'work/research')]
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import require,record
from asr18_owned_source_closure_v1 import owned_closure

def main():
    cpu=A/'CPU_frontend_actual_02';guard=A/'CPU_frontend_guard_02/closure.json'
    require(json.loads((cpu/'report.json').read_text())['passed'] and json.loads(guard.read_text())['actual_exit_code']==0,'Actual successful CPU fixture')
    write_json(cpu/'actual_owned_CLOSED0.json',{'kind':'ASR18_owned_actual_tool_closure_v1','owner':'/root/memory_diagnosis','session':14533,'native_PID':24404,
        'actual_exit_code':0,'actual_tool_chunk_id':'ba26fc','report':record(cpu/'report.json'),'native_closure':record(guard),
        'original_v1_failed_late_module_classification_preserved':record(A/'CPU_frontend_guard_01/closure.json')})
    path=P/'work/research/run_ASR18_component_profile_v1.py'
    driver=(P/'work/research/run_ASR18_real_TDT_preparation_v10.py').read_text().replace('from asr18_full_hard_2D_native_v1 import validate','from asr18_component_memory_profile_v1 import validate')
    driver=driver.replace("'max_native_updates':2","'max_native_updates':0").replace("choices=['fullhard'],default='fullhard'","choices=['profile'],default='profile'")
    driver=driver.replace("dependencies.append(Path(a.review))","""dependencies.extend([SOURCE/'asr18_component_memory_profile_v1.py',AREA/'CPU_frontend_actual_02/report.json',AREA/'CPU_frontend_actual_02/actual_owned_CLOSED0.json',AREA/'CPU_frontend_guard_02/closure.json'])
    dependencies.extend(owned_closure(dependencies));dependencies=sorted(set(dependencies))
    dependencies.append(Path(a.review))""")
    ast.parse(driver)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(driver)
    bindings=[record(x) for x in owned_closure([path,S/'asr18_component_memory_profile_v1.py'])]
    bindings.extend(record(x) for x in (cpu/'report.json',cpu/'actual_owned_CLOSED0.json',guard))
    manifest={'kind':'ASR18_component_profile_source_manifest_v1','bindings':bindings,'max_native_updates':0,
        'synchronous_component_hooks':True,'full_T1991_U574_gradient_only':True,'CPU_frontend_byte_comparison_full_2_rows':True,'production_approval_claimed':False}
    write_json(A/'component_profile_source_manifest_v1.json',manifest)
    write_json(A/'component_profile_bounded_local_review_v1.json',{'scope':'bounded_real_source_TDT_validation','accepted':True,'production_approval_claimed':False,'bindings':bindings,
        'source_manifest':record(A/'component_profile_source_manifest_v1.json'),'limits':['Observational synchronized hooks, not throughput benchmark','0Adam/original full source clocks','Exact CPU versus original parent feature/mask equality required','No actual memory remedy claim before measurement']})
    print(json.dumps(record(A/'component_profile_source_manifest_v1.json')))

if __name__=='__main__':main()
