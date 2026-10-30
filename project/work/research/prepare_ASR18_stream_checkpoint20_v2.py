"""New measured-GC/new2D same-process diagnostic; old failed attempt preserved."""
import ast,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
sys.path[:0]=[str(S),str(P/'work/research')]
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from asr18_owned_source_closure_v1 import owned_closure

def create(path,text):
    ast.parse(text)
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(text)

def main():
    hard=A/'full_hard_2D_native_01'
    require(json.loads((hard/'report.json').read_text())['passed'],'Actual full hard native closure required')
    write_json(hard/'actual_owned_CLOSED0.json',{'kind':'ASR18_owned_actual_tool_closure_v1','owner':'/root/memory_diagnosis','session':85436,
        'native_PID':27544,'actual_exit_code':0,'actual_tool_chunk_id':'4790ca','report':record(hard/'report.json'),
        'native_and_state_report':record(hard/'full_hard_native_validation.json'),'native_updates':2,'diagnostic_only':True})
    config=json.loads((S/'bpe_broad_stream_checkpoint20_v1.disabled.json').read_text())
    config.update(kind='ASR18_sameprocess20_fixed14_streaming_checkpoint_diagnostic_v2',production_enabled=False,
        new_joint_policy='tdt_joint_canonical_2d_fp32_v2.py',native_loss='tdt_tiled_full_native_v6.py',
        caller='bpe_broad_production_v6.py',checkpoint_cadence=[1,10,20],
        oversized_geometry_policy='T1_U128_full8711_normalization_full_native_TU_DP; unresolved_bounds_stop_before_draw_no_exclusion',
        graph_boundary_GC='after each finished backward graph deletion; RNG and parameter-gradient pointers/versions invariant',
        material_workload_changes=['new explicit canonicalT1/U128 FP32 policy','instance-only denominator binding','measured perclip graph-boundary GC','streamed ACCESS_READ checkpoint restore with owned CPU steps'],
        previous_failed20_attempt=record(A/'sameprocess_stream_checkpoint20_01/report.json'),
        actual_full_long_geometry_proof=record(hard/'actual_owned_CLOSED0.json'))
    config_path=S/'bpe_broad_stream_checkpoint20_v2.disabled.json';require(not config_path.exists(),'Fresh config path')
    write_json(config_path,config)
    probe=(S/'asr18_broad_stream_checkpoint20_probe_v1.py').read_text().replace('from bpe_broad_production_v5 import *','from bpe_broad_production_v6 import *')
    probe=probe.replace('real_target_TDT_gradients_06/real_model_equivalence.json','real_target_2D_TDT_gradients_01/real_model_equivalence.json')
    probe=probe.replace('real_target_TDT_large_02/full_large_row_validation.json','full_hard_2D_native_01/full_hard_native_validation.json')
    probe=probe.replace("'bpe_broad_production_v5.py'","'bpe_broad_production_v6.py'")
    probe=probe.replace("'ASR18_sameprocess_14source20update_streaming_checkpoint_diagnostic_v1'","'ASR18_sameprocess_14source20update_streaming_checkpoint_diagnostic_v2'")
    needle="    try:\n        for update in range(1,21):"
    replacement="""    def publish_checkpoint(binding,state):
        require(record(binding['path'])==binding,'Completed checkpoint metadata binding differs')
        document=json.loads(Path(binding['path']).read_text())
        require(record(document['payload']['path'])==document['payload'],'Completed checkpoint payload binding differs')
        write_json(folder/'latest_coherent_checkpoint.json',{'checkpoint':binding,'payload':document['payload'],'sampler':state,
            'clocks':[c+optimizer.updates for c in (64250,2000,62250)],'writer_returned_after_fsync':True,'diagnostic_only':True})
    try:
        for update in range(1,21):"""
    require(probe.count(needle)==1,'Exact loop insertion required');probe=probe.replace(needle,replacement)
    probe=probe.replace("                old_prefix=journal.binding();", "                publish_checkpoint(checkpoint1,state)\n                old_prefix=journal.binding();")
    probe=probe.replace("        applied=sum(e['augmentation']", "        publish_checkpoint(saved,state)\n        applied=sum(e['augmentation']")
    create(S/'asr18_broad_stream_checkpoint20_probe_v2.py',probe)
    driver=(P/'work/research/run_ASR18_stream_checkpoint20_v1.py').read_text()
    driver=driver.replace('asr18_broad_stream_checkpoint20_probe_v1','asr18_broad_stream_checkpoint20_probe_v2')
    driver=driver.replace('bpe_broad_stream_checkpoint20_v1.disabled.json','bpe_broad_stream_checkpoint20_v2.disabled.json')
    driver=driver.replace('tdt_joint_canonical_fp32_v1','tdt_joint_canonical_2d_fp32_v2').replace('tdt_tiled_full_native_v4','tdt_tiled_full_native_v6')
    driver=driver.replace("dependencies.append(Path(a.review))","""dependencies.extend([SOURCE/'bpe_broad_production_v6.py',SOURCE/'tdt_exact_denominator_instance_v1.py',AREA/'full_hard_2D_native_01/actual_owned_CLOSED0.json',AREA/'full_hard_2D_native_01/full_hard_native_validation.json',AREA/'real_target_2D_TDT_gradients_01/real_model_equivalence.json',AREA/'instance_denominator_joint_01/actual_joint_precision.json'])
    dependencies.extend(owned_closure(dependencies));dependencies=sorted(set(dependencies))
    dependencies.append(Path(a.review))""")
    driver_path=P/'work/research/run_ASR18_stream_checkpoint20_v2.py';create(driver_path,driver)
    entries=[driver_path,S/'asr18_broad_stream_checkpoint20_probe_v2.py',S/'bpe_broad_production_v6.py',S/'tdt_tiled_full_native_v6.py']
    bindings=[record(x) for x in owned_closure(entries)]
    bindings.extend(record(x) for x in (config_path,hard/'actual_owned_CLOSED0.json',hard/'full_hard_native_validation.json',
        A/'real_target_2D_TDT_gradients_01/real_model_equivalence.json',A/'instance_denominator_joint_01/actual_joint_precision.json'))
    manifest={'kind':'ASR18_sameprocess20_changed_source_manifest_v2','bindings':bindings,'max_native_updates':20,'same_process':True,
        'fixed_14_TR_repeated_fixture':True,'full_corpus_passes':0,'original_clocks':[64250,2000,62250],
        'checkpoint_cadence':[1,10,20],'actual_restore_after':[1,10],'production_enabled':False,
        'previous_failed20_preserved':config['previous_failed20_attempt'],'material_workload_changes':config['material_workload_changes']}
    write_json(A/'production_caller_source_review_manifest_v11.json',manifest)
    write_json(A/'sameprocess20_v2_bounded_local_review.json',{'scope':'bounded_production_caller_preparation','accepted':True,
        'production_approval_claimed':False,'bindings':bindings,'source_manifest':record(A/'production_caller_source_review_manifest_v11.json'),
        'limits':['Same fixed14 source geometry and carried replay only','No tolerance/guard/margin reduction','No production mean or corpus coverage','Unqualified checkpoints only']})
    print(json.dumps({'manifest':record(A/'production_caller_source_review_manifest_v11.json'),'owned_and_metadata_bindings':len(bindings)}))

if __name__=='__main__':main()
