"""Generate a fresh bounded 2D all207 oracle; never rewrite admitted revisions."""
import ast,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path[:0]=[str(S),str(P/'work/research')]
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from asr18_owned_source_closure_v1 import owned_closure

def create(path,text):
    ast.parse(text)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)

def main():
    helper=(S/'asr18_real_target_TDT_validation_v6.py').read_text()
    helper=helper[:helper.index('\ndef large(')]
    helper=helper.replace('from tdt_tiled_full_native_v4 import bounded_tdt_loss','from tdt_tiled_full_native_v5 import bounded_tdt_loss')
    helper=helper.replace("source=AREA/'new_broad_caller_native_03/decoded/report.json'","source=AREA/'hard_U_rows_decode_01/decoded/report.json'")
    helper=helper.replace("'kind':'real_source_all207_TDT_gradient_equivalence_v1'","'kind':'real_source_2D_all207_TDT_gradient_equivalence_v1'")
    helper=helper.replace("'gradient_tiling_BF16_rounding_not_byte_equality':True","'gradient_tiling_BF16_rounding_not_byte_equality':False")
    helper=helper.replace("'tdt_joint_canonical_fp32_v1'","'tdt_joint_canonical_2d_fp32_v2'").replace("'tdt_tiled_full_native_v4'","'tdt_tiled_full_native_v5'")
    helper=helper.replace("'raw_labels_and_full_audio':True,","'raw_labels_and_full_audio':True,'loss_acoustic_frames':15,'acoustic_subset_numerical_oracle_only':True,'full_length_objective_fit_claimed':False,")
    helper=helper.replace("    dense=partial(chunked_tdt_loss,sigma=0.,omega=0.,recompute=True,max_joint_bytes=1_700_000_000)\n    bounded=partial(bounded_tdt_loss,guard=stop_check)","""    def subset(function):
        def call(*args,**kwargs):
            lengths=kwargs['logit_lengths'].clone()
            require(int(lengths[0])>=15,'Actual full source must exceed oracle subset')
            kwargs['logit_lengths']=lengths.clamp(max=15)
            return function(*args,**kwargs)
        return call
    dense=subset(partial(chunked_tdt_loss,sigma=0.,omega=0.,recompute=True,max_joint_bytes=1_700_000_000))
    bounded=subset(partial(bounded_tdt_loss,guard=stop_check))""")
    helper=helper.replace("'T':row['encoder_frames']","'T':15")
    helper=helper.replace("'joint_fixture':record(folder/('actual_joint_fixture_'+str(index)+'.pt'))","'all207_byte_equal':all(v['max_abs']==0. for v in metrics),'loss_acoustic_frames':15,'full_input_encoder_frames':row['encoder_frames'],'joint_fixture':record(folder/('actual_joint_fixture_'+str(index)+'.pt'))")
    helper=helper.replace("actual_joint_precision_08/actual_joint_precision.json","canonical2D_joint_GC_01/actual_joint_precision.json")
    create(S/'asr18_real_target_TDT_validation_v9.py',helper)
    driver=(P/'work/research/run_ASR18_real_TDT_preparation_v6.py').read_text()
    driver=driver.replace('asr18_real_target_TDT_validation_v6','asr18_real_target_TDT_validation_v9')
    driver=driver.replace('from asr18_real_target_TDT_validation_v9 import compare,large','from asr18_real_target_TDT_validation_v9 import compare')
    driver=driver.replace("(compare if a.validation_mode=='gradients' else large)","compare")
    driver=driver.replace('tdt_joint_canonical_fp32_v1','tdt_joint_canonical_2d_fp32_v2')
    driver=driver.replace('tdt_tiled_full_native_v4','tdt_tiled_full_native_v5')
    driver=driver.replace('actual_joint_precision_08/actual_joint_precision.json','canonical2D_joint_GC_01/actual_joint_precision.json')
    driver=driver.replace("AREA/'new_broad_caller_native_03/decoded/report.json'","AREA/'hard_U_rows_decode_01/decoded/report.json'")
    driver=driver.replace("dependencies.append(Path(a.review))","""dependencies.extend([AREA/'hard_U_rows_decode_01/actual_owned_CLOSED0.json',AREA/'hard_U_rows_decode_01/guard/closure.json'])
    source_PCM=read_json(AREA/'hard_U_rows_decode_01/decoded/report.json')
    require(source_PCM['passed'] and [v['row']['target_tokens'] for v in source_PCM['rows']]==[574,484],'Actual complete retained hard rows')
    dependencies.extend(Path(v['PCM_cache']['path']) for v in source_PCM['rows'])
    from asr18_owned_source_closure_v1 import owned_closure
    dependencies.append(PROJECT/'work/research/asr18_owned_source_closure_v1.py')
    dependencies.extend(owned_closure(dependencies));dependencies=sorted(set(dependencies))
    dependencies.append(Path(a.review))""")
    driver=driver.replace("choices=['gradients','large']","choices=['gradients']")
    driver=driver.replace("'max_native_updates':1 if a.validation_mode=='large' else 0","'max_native_updates':0")
    create(P/'work/research/run_ASR18_real_TDT_preparation_v9.py',driver)
    decode=A/'hard_U_rows_decode_01'
    closure=json.loads((decode/'guard/closure.json').read_text());require(closure['actual_exit_code']==0,'Actual native CPU closure')
    write_json(decode/'actual_owned_CLOSED0.json',{'kind':'ASR18_owned_actual_tool_closure_v1','owner':'/root/memory_diagnosis','session':6450,'native_PID':37816,'actual_exit_code':0,'actual_tool_chunk_id':'8e3c82','report':record(decode/'decoded/report.json'),'native_closure':record(decode/'guard/closure.json'),'full_audio_and_targets_retained':True,'draws_consumed':0})
    entries=[P/'work/research/run_ASR18_real_TDT_preparation_v9.py',S/'asr18_real_target_TDT_validation_v9.py',S/'tdt_joint_canonical_2d_fp32_v2.py',S/'tdt_tiled_full_native_v5.py']
    bindings=[record(x) for x in owned_closure(entries)]
    bindings.extend(record(x) for x in [decode/'actual_owned_CLOSED0.json',decode/'decoded/report.json',decode/'guard/closure.json',A/'canonical2D_joint_GC_01/actual_joint_precision.json'])
    manifest={'kind':'ASR18_2D_full_model_bounded_source_review_v1','bindings':bindings,'optimizer_updates':0,'full159s_120s_input':True,'loss_acoustic_frame_subset':15,'full_U574_U484_targets':True,'same_explicit_dense_and_tiled_numerical_policy':True,'full_length_objective_fit_claimed':False,'production_approval_claimed':False}
    write_json(A/'candidate_2D_full_model_source_review_v1.json',manifest)
    write_json(A/'candidate_2D_full_model_bounded_local_review_v1.json',{'scope':'bounded_real_source_TDT_validation','accepted':True,'production_approval_claimed':False,'bindings':bindings,'source_manifest':record(A/'candidate_2D_full_model_source_review_v1.json'),'limitations':['15 acoustic loss frames only; all source audio enters encoder and all targets retained','No Adam or full-length fit claim','ROOT final production review remains required']})
    print(json.dumps({'manifest':record(A/'candidate_2D_full_model_source_review_v1.json'),'review':record(A/'candidate_2D_full_model_bounded_local_review_v1.json'),'owned_sources':len(bindings)}))

if __name__=='__main__':main()
