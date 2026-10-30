from pathlib import Path
import ast, hashlib

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'outputs/parakeet_turkish'
parent=HERE/'run_TSC_uniform_full_epoch_v1.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='1943266496b654bc3439c0f667f8a197bb620a0c1d110d5bd9269cb26910d885'
t=parent.read_text(encoding='utf-8')
t=t.replace('train_lora_batched_tsc_uniform_full_epoch_v1.py','train_lora_batched_tsc_uniform_io_v1.py')
t=t.replace('tsc_uniform_full_epoch_transition_v1','tsc_uniform_io_transition_v1')
t=t.replace('tsc_uniform_majority_ptfocus_sampler_v1.py','tsc_uniform_io_sampler_v1.py')
t=t.replace('guarded_TSC_8199_uniform_majority_ptfocus_fullpass_controller_v1','guarded_TSC_8199_uniform_IO_control_controller_v1')
t=t.replace("WORKER_SHA='1c33cbde6933c55d0839d0ebf82253fb3b673dde7f5a105adc3905e14f7013c3'",
            "WORKER_SHA='"+hashlib.sha256((HERE/'train_lora_batched_tsc_uniform_io_v1.py').read_bytes()).hexdigest()+"'")
t=t.replace("TRANSITION_SHA='7cf6aaa78ef3372353cd3320aa606add08927dbecca17a3e91dfd482474b3ce3'",
            "TRANSITION_SHA='ac7ddf99598cb5a1cf5e2c904c55f21564d596d1370a643ea781697e221443fe'")
t=t.replace("SAMPLER_SHA='61f0c253c09abe49a6dbbf47f6ab4cbf3b0d6fdbf75d6f4490d59f2f381c2a2e'",
            "SAMPLER_SHA='b1a500400ef4bec5af8df984c82653df60f9e08306c7db758eeaa2c03c709d30'")
t=t.replace('EXPECTED_SYNC_LINE=748', 'EXPECTED_SYNC_LINE=775')
t=t.replace("'native_tdt_checkpointed_v4.py':ADAPTER_SHA,'compact_training_row_objects_v1.py':COMPACTOR_SHA,",
            "'native_tdt_checkpointed_v4.py':ADAPTER_SHA,'compact_training_row_objects_v1.py':COMPACTOR_SHA,\n        'atomic_paired_history_v1.py':'c0d18921d1a268ba69cbae1e2ad8efde657165450b2551d2297e7fba5365d8dd',")
marker="    values['retention_stop_policy']=option(argv,'--retention-stop-policy')"
assert t.count(marker)==1
t=t.replace(marker,marker+"\n    values['persistence_mode']=option(argv,'--persistence-mode')\n    values['persistence_benchmark_updates']=int(option(argv,'--persistence-benchmark-updates'))")
marker="    state_path=scoped(option(argv,'--resume-exact' if exact else '--resume-majority'))"
assert t.count(marker)==1
t=t.replace(marker,"    require(option(argv,'--persistence-mode')==context['receipt']['persistence_mode']\n        and option(argv,'--persistence-benchmark-updates')=='100','Matched technical persistence mode/budget differs')\n"+marker)
old="('--native-backend-provenance',paths['native']),('--stop-file',paths['stop']),('--max-wall-seconds',args.window_hours*3600-300)):"
assert t.count(old)==1
t=t.replace(old,"('--persistence-mode',args.persistence_mode),('--persistence-benchmark-updates',100),\n        "+old)
old="        'PT_focus_boundary_step':context['PT_focus_boundary_step'],'prior_uniform_draws':context['prior_uniform_draws'],"
assert t.count(old)==1
t=t.replace(old,old+"\n        'persistence_mode':args.persistence_mode,'technical_control_updates':100,\n        'source_lineage_end_step':25109,'technical_end_step':25209,\n        'isolated_same_volume_publications':True,'production_mirror_reader_contention_measured':False,")
marker="    proof=read(run/'majority_initial_source_equivalence.json')"
assert t.count(marker)==1
t=t.replace(marker,"    require(config['phase']=='HF_TSC_8199_uniform_IO_control_v1'\n        and selection['persistence_control_updates']==100\n        and selection['persistence_mode']==plan['persistence_mode']\n        and selection['completed_steps']==25209,'Actual complete matched100-update technical control required')\n"+marker)
marker="        'published_step':state.get('published_step',state['resume_step']),"
assert t.count(marker)==1
t=t.replace(marker,marker+"\n        'persistence_mode':state['persistence_mode'],'technical_control_updates':100,\n        'actual_persistence_control_updates':max(0,state.get('published_step',state['resume_step'])-25109),\n        'original_full_pass_already_complete':True,")
marker="    p.add_argument('--indexed-tr-directory',type=Path)"
assert t.count(marker)==1
t=t.replace(marker,marker+"\n    p.add_argument('--persistence-mode',choices=('double_json','single_serialize_two_commits'),required=True)")
ast.parse(t)
target=HERE/'run_TSC_uniform_io_v1.py'
assert not target.exists()
target.write_text(t,encoding='utf-8')
print({'controller':str(target),'sha':hashlib.sha256(target.read_bytes()).hexdigest()})
