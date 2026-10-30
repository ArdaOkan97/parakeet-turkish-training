"""Pin exactly two complete long-row diagnostic updates and streaming restores."""
import ast,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path[:0]=[str(S),str(P/'work/research')]
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from asr18_owned_source_closure_v1 import owned_closure

def main():
    child=A/'instance_denominator_joint_01'
    require(json.loads((child/'report.json').read_text())['passed'],'Actual instance oracle closed cleanly')
    write_json(child/'actual_owned_CLOSED0.json',{'kind':'ASR18_owned_actual_tool_closure_v1','owner':'/root/memory_diagnosis','session':86233,'native_PID':32036,
        'actual_exit_code':0,'actual_tool_chunk_id':'744c20','report':record(child/'report.json'),'math_report':record(child/'actual_joint_precision.json'),'optimizer_updates':0})
    first=A/'real_target_2D_TDT_gradients_01'
    write_json(first/'actual_owned_CLOSED0.json',{'kind':'ASR18_owned_actual_tool_closure_v1','owner':'/root/memory_diagnosis','session':12356,'native_PID':31512,
        'actual_exit_code':0,'actual_tool_chunk_id':'0fffdc','report':record(first/'report.json'),'math_report':record(first/'real_model_equivalence.json'),
        'optimizer_updates':0,'all207_byte_exact':False,'unchanged_predeclared_tolerance_gates_passed':True,'loss_acoustic_subset_frames':15})
    path=P/'work/research/run_ASR18_real_TDT_preparation_v10.py'
    driver=(P/'work/research/run_ASR18_real_TDT_preparation_v9.py').read_text()
    driver=driver.replace('from asr18_real_target_TDT_validation_v9 import compare','from asr18_full_hard_2D_native_v1 import validate')
    driver=driver.replace('result=compare(','result=validate(')
    driver=driver.replace("'max_native_updates':0","'max_native_updates':2")
    driver=driver.replace("choices=['gradients'],default='gradients'","choices=['fullhard'],default='fullhard'")
    driver=driver.replace("dependencies.append(Path(a.review))","""dependencies.extend([SOURCE/'asr18_full_hard_2D_native_v1.py',SOURCE/'tdt_tiled_full_native_v6.py',SOURCE/'tdt_exact_denominator_instance_v1.py',SOURCE/'bpe_streaming_training_state_v1.py',SOURCE/'bpe_streaming_checkpoint_v1.py',AREA/'real_target_2D_TDT_gradients_01/real_model_equivalence.json',AREA/'real_target_2D_TDT_gradients_01/actual_owned_CLOSED0.json',AREA/'instance_denominator_joint_01/actual_joint_precision.json',AREA/'instance_denominator_joint_01/actual_owned_CLOSED0.json'])
    dependencies.extend(owned_closure(dependencies));dependencies=sorted(set(dependencies))
    dependencies.append(Path(a.review))""")
    ast.parse(driver)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(driver)
    entries=[path,S/'asr18_full_hard_2D_native_v1.py',S/'tdt_joint_canonical_2d_fp32_v2.py',S/'tdt_tiled_full_native_v6.py']
    bindings=[record(x) for x in owned_closure(entries)]
    bindings.extend(record(x) for x in (first/'real_model_equivalence.json',first/'actual_owned_CLOSED0.json',child/'actual_joint_precision.json',child/'actual_owned_CLOSED0.json',A/'hard_U_rows_decode_01/decoded/report.json'))
    manifest={'kind':'ASR18_full_hard_2D_source_review_manifest_v1','bindings':bindings,'native_updates_bound':2,
        'full159s_U574_and120s_U484':True,'all_labels_and_audio_retained':True,'streaming_checkpoint_and_restore_after_each':True,
        'production_approval_claimed':False,'unqualified_diagnostic_only':True}
    write_json(A/'full_hard_2D_source_review_manifest_v1.json',manifest)
    write_json(A/'full_hard_2D_bounded_local_review_v1.json',{'scope':'bounded_real_source_TDT_validation','accepted':True,'production_approval_claimed':False,
        'bindings':bindings,'source_manifest':record(A/'full_hard_2D_source_review_manifest_v1.json'),'limits':['Exactly two full pinned retained rows; no production sampler/normalization','No augmentation or replay in this long-row geometry diagnostic','No sustained20 update or quality/winner claim','Fresh unchanged resource admission and no cap relaxation']})
    print(json.dumps({'manifest':record(A/'full_hard_2D_source_review_manifest_v1.json'),'bindings':len(bindings)}))

if __name__=='__main__':main()
