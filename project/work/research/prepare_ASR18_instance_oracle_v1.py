"""Freeze a finite captured-joint test of the explicit wrapper lifetime change."""
import ast,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path[:0]=[str(S),str(P/'work/research')]
from bpe_broad_audio_v1 import record
from atomic_io_retry_v1 import write_json
from asr18_owned_source_closure_v1 import owned_closure

def main():
    path=P/'work/research/run_ASR18_actual_joint_precision_v11.py'
    driver=(P/'work/research/run_ASR18_actual_joint_precision_v10.py').read_text().replace('asr18_actual_joint_precision_v10','asr18_actual_joint_precision_v11')
    driver=driver.replace("AREA/'real_target_TDT_gradients_02/actual_joint_fixture_0.pt',AREA/'real_target_TDT_gradients_02/actual_joint_fixture_1.pt'","AREA/'real_target_2D_TDT_gradients_01/actual_joint_fixture_0.pt',AREA/'real_target_2D_TDT_gradients_01/actual_joint_fixture_1.pt'")
    driver=driver.replace("dependencies.append(Path(a.review))","dependencies.extend([SOURCE/'tdt_tiled_full_native_v6.py',SOURCE/'tdt_exact_denominator_instance_v1.py',AREA/'real_target_2D_TDT_gradients_01/real_model_equivalence.json',AREA/'candidate_instance_denominator_delta_v1.json'])\n    dependencies=sorted(set(dependencies))\n    dependencies.append(Path(a.review))")
    ast.parse(driver)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(driver)
    entries=[path,S/'asr18_actual_joint_precision_v11.py',S/'tdt_tiled_full_native_v6.py']
    bindings=[record(x) for x in owned_closure(entries)]
    bindings.extend(record(x) for x in (A/'real_target_2D_TDT_gradients_01/real_model_equivalence.json',A/'candidate_instance_denominator_delta_v1.json'))
    value={'kind':'ASR18_instance_denominator_source_review_manifest_v1','bindings':bindings,'optimizer_updates':0,'full_ASR_model_loaded':False,'production_approval_claimed':False}
    write_json(A/'instance_denominator_source_review_manifest_v1.json',value)
    write_json(A/'instance_denominator_bounded_local_review_v1.json',{'scope':'bounded_actual_joint_precision_diagnosis','accepted':True,'production_approval_claimed':False,
        'bindings':bindings,'manifest':record(A/'instance_denominator_source_review_manifest_v1.json'),'limits':['Captured actual15frame U574/U484 only','Lifetime arms keep head/inputs alive','No actual training-phase memory benefit claim']})
    print(json.dumps(record(A/'instance_denominator_source_review_manifest_v1.json')))

if __name__=='__main__':main()
