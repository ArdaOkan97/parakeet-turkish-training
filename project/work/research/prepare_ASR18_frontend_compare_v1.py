"""Freeze a cheap no-model CUDA-parent frontend equality control."""
import ast,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path[:0]=[str(S),str(P/'work/research')]
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record
from asr18_owned_source_closure_v1 import owned_closure

def main():
    text=(P/'work/research/run_ASR18_actual_joint_precision_v11.py').read_text()
    first=text.index('    from native_tdt_minimal_import_v1 import prepare')
    last=text.index('    def stop():',first)
    text=text[:first]+"    emit('original_parent_frontend_runtime_ready',coherent=coherent,cap=cap)\n"+text[last:]
    text=text.replace('from asr18_actual_joint_precision_v11 import validate','from asr18_parent_frontend_compare_v1 import validate')
    text=text.replace('from asr18_memory_diagnosis_v1 import storage','from asr18_memory_diagnosis_v2 import storage')
    text=text.replace("'bounded_actual_joint_precision_diagnosis'","'bounded_original_frontend_precision_control'")
    text=text.replace("dependencies.append(Path(a.review))","""dependencies.extend([SOURCE/'asr18_parent_frontend_compare_v1.py',SOURCE/'asr18_token_only_carrier_v1.py',AREA/'CPU_frontend_actual_02/report.json',AREA/'CPU_frontend_actual_02/actual_owned_CLOSED0.json',PROJECT/'work/research/asr18_memory_diagnosis_v2.py'])
    dependencies.extend(owned_closure(dependencies));dependencies=sorted(set(dependencies))
    dependencies.append(Path(a.review))""")
    ast.parse(text);path=P/'work/research/run_ASR18_frontend_compare_v1.py'
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    bindings=[record(x) for x in owned_closure([path,S/'asr18_parent_frontend_compare_v1.py',S/'asr18_token_only_carrier_v1.py'])]
    bindings.extend(record(x) for x in (A/'CPU_frontend_actual_02/report.json',A/'CPU_frontend_actual_02/actual_owned_CLOSED0.json'))
    write_json(A/'frontend_compare_source_manifest_v1.json',{'kind':'ASR18_original_frontend_precision_control_source_manifest_v1','bindings':bindings,'ASR_model_loaded':False,'native_updates':0,'production_approval_claimed':False})
    write_json(A/'frontend_compare_bounded_local_review_v1.json',{'scope':'bounded_original_frontend_precision_control','accepted':True,'bindings':bindings,
        'production_approval_claimed':False,'source_manifest':record(A/'frontend_compare_source_manifest_v1.json'),'limits':['Exact2full source clips only','NoASR/model/Adam loads','Every feature/mask byte plus original tokenizer/blank identity must match','No steady resource improvement claim']})
    print(json.dumps(record(A/'frontend_compare_source_manifest_v1.json')))

if __name__=='__main__':main()
