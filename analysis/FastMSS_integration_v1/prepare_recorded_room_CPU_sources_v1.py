"""Stdlib source-only assembly of explicit new policy/protocol/workers/controller."""
import ast,copy,hashlib,json
from pathlib import Path
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
F=Path(r'E:/gpu_training_scratchpad/analysis/FastMSS_integration_v1')
A=Path(r'E:/gpu_training_scratchpad/analysis/ASR18_memory_diagnosis_v1')
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
def source(path,text):
    ast.parse(text)
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def change(text,old,new):
    assert old in text,old
    return text.replace(old,new)

protocol=S/'fastmss_recorded_room_feature_protocol_v1.py'
reference=R/'produce_FastMSS_recorded_room_reference_CPU_v1.py'
frontend=R/'produce_FastMSS_recorded_room_frontend_CPU_v1.py'
controller=R/'run_ASR18_recorded_room_CPU_window_v1.py'
adapter=S/'fastmss_recorded_room_train_v1.py'
for path in (protocol,reference,frontend,controller):assert not path.exists(),path

t=(S/'fastmss_window_aggregate_feature_protocol_v1.py').read_text()
t=change(t,'import fastmss_owned_process_closure_v2 as process','import fastmss_owned_process_closure_v2 as process\nimport fastmss_recorded_room_train_v1 as augmentation')
t=t.replace('ASR18_aggregate_CPU_window_validation_v1','ASR18_recorded_room_CPU_benchmark_v1')
t=t.replace('FastMSS_window_scoped_CPU_source_and_numerical_contract_v1','FastMSS_recorded_room_CPU_source_and_numerical_contract_v1')
t=t.replace('FastMSS_window_scoped_CPU_preparation_request_v1','FastMSS_recorded_room_CPU_preparation_request_v1')
t=t.replace('isolated_original_window_public_protocol_validation','recorded_room_first4_native_benchmark_only')
t=t.replace('FastMSS_owned_aggregate_CPU_feature_window_v1','FastMSS_owned_recorded_room_CPU_feature_window_v1')
t=t.replace('FastMSS_independent_full_CPU_reference_window_v2','FastMSS_recorded_room_independent_CPU_reference_window_v1')
t=t.replace('FastMSS_reference_update_within_actual_CPU_window_v2','FastMSS_recorded_room_reference_update_within_actual_CPU_window_v1')
t=t.replace('FastMSS_original_full_CPU_frontend_window_v2','FastMSS_recorded_room_full_CPU_frontend_window_v1')
t=t.replace('FastMSS_provisional_update_within_actual_frontend_window_v1','FastMSS_recorded_room_update_within_actual_frontend_window_v1')
t=t.replace('FastMSS_actual_aggregate_CPU_frontend_window_closure_v1','FastMSS_actual_recorded_room_CPU_frontend_window_closure_v1')
t=t.replace('FastMSS_actual_aggregate_closed_CPU_feature_window_v1','FastMSS_actual_recorded_room_closed_CPU_feature_window_v1')
t=t.replace('produce_FastMSS_reference_window_CPU_v2.py',reference.name).replace('produce_FastMSS_frontend_window_CPU_v2.py',frontend.name)
t=t.replace('run_ASR18_public_CPU_window_validation_v1.py',controller.name)
t=change(t,"and value['production_source_eligible'] is False and value['validation_only'] is True,",
    "and value['production_source_eligible'] is False and value['validation_only'] is False\n        and value['native_benchmark_only'] is True,")
t=change(t,"'process': Path(process.__file__)}", "'process': Path(process.__file__), 'augmentation_adapter': Path(augmentation.__file__),\n"
    "        'real_noise_mixer': Path(augmentation.mixer.__file__), 'real_noise_decoder': Path(augmentation.decoder.__file__),\n"
    "        'room_provenance': Path(augmentation.rooms.__file__)}")
t=change(t,"    return value, plan\n\n\ndef _check_reference_rows", "    _, identity = augmentation.policy_identity(value['augmentation_policy'], guard)\n"
    "    require(same(identity, value['augmentation_identity']) and value['original_augmentation_identity'] == plan['augmentation_identity']\n"
    "        and plan['phase'] == bridge.WEAK and len(plan['updates']) == 4\n"
    "        and sum(len(v['rows']) for v in plan['updates']) == 64,\n"
    "        'Explicit new augmentation overlay; original schedule/session/rows remain immutable')\n"
    "    return value, plan\n\n\ndef _check_reference_rows")
t=change(t,"'capacity_bytes': CAPACITY, 'augmentation_identity': plan['augmentation_identity'], 'frontend': frontend,",
    "'capacity_bytes': CAPACITY, 'augmentation_identity': contract['augmentation_identity'], 'frontend': frontend,\n"
    "        'original_augmentation_identity': plan['augmentation_identity'], 'native_benchmark_only': True,")
t=change(t,"and same(owner['augmentation_identity'], plan['augmentation_identity']), 'Owner/full original bridge differs')",
    "and same(owner['original_augmentation_identity'], plan['augmentation_identity'])\n"
    "        and same(owner['augmentation_identity'], read_bound(owner['source_contract'], guard)['augmentation_identity'])\n"
    "        and owner['native_benchmark_only'] is True, 'Owner/original bridge and explicit new augmentation overlay differ')")
t=t.replace('Exact permanently inactive validation-only source contract','Exact production-disabled native-benchmark source contract')
t=t.replace('Exact fresh validation-only aggregate owner/source/namespace/caps','Exact fresh native-benchmark aggregate owner/source/namespace/caps')
t=t.replace('Preserve any outstanding NEW validation window','Preserve any outstanding NEW recorded-room window')
t=t.replace('"""Inactive finite window CPU protocol: TWO real aggregate leaves, never N fake exits.',
    '"""NEW native-readable recorded-room benchmark: TWO real aggregate leaves.')
t=t.replace('This explicit validation namespace cannot adopt or retire the existing shared',
    'This explicit benchmark namespace cannot adopt the existing shared')
source(protocol,t)

t=(R/'produce_FastMSS_reference_window_CPU_v2.py').read_text()
t=t.replace('fastmss_window_aggregate_feature_protocol_v1','fastmss_recorded_room_feature_protocol_v1')
t=t.replace('inactive validation slot','production-disabled recorded-room benchmark slot')
source(reference,t)

t=(R/'produce_FastMSS_frontend_window_CPU_v2.py').read_text()
t=t.replace('produce_FastMSS_reference_window_CPU_v2','produce_FastMSS_recorded_room_reference_CPU_v1')
t=t.replace('fastmss_window_aggregate_feature_protocol_v1','fastmss_recorded_room_feature_protocol_v1')
t=change(t,"            identity = core.bridge.frozen_augmentation(plan['augmentation_identity'], guard)",
    "            import fastmss_recorded_room_train_v1 as augmentation\n"
    "            _, identity = augmentation.policy_identity(source['augmentation_policy'], guard)")
t=change(t,"identity['numpy_version'] and scipy.__version__ == identity['scipy_version']",
    "source['numerical_runtime_versions']['NumPy'] and scipy.__version__ == source['numerical_runtime_versions']['SciPy']")
start=t.index('            weak_transform = None\n');end=t.index("            active = {'index':",start)
t=t[:start]+'''            augmenter = augmentation.RecordedRoomTrainingAdapter(source['augmentation_policy'], reader=reader,
                guard=guard, ROOT_review=request['ROOT_review'], configuration=request['configuration'])
            core.require(core.same(augmenter.identity, source['augmentation_identity']), 'Exact new reviewed draw policy')
            def weak_transform(wave, row, context):
                with timings.stage('recorded_noise_and_16room_training_transform'):
                    return augmenter.transform(wave, row, context)
'''+t[end:]
t=t.replace("'actual_full_CPU_preparation_speedup_claimed': False, 'production_enabled': False}",
    "'actual_full_CPU_preparation_speedup_claimed': False, 'production_enabled': False,\n"
    "        'augmentation_identity': identity, 'native_benchmark_only': True}")
source(frontend,t)

t=(R/'run_ASR18_public_CPU_window_validation_v1.py').read_text()
t=t.replace('produce_FastMSS_reference_window_CPU_v2','produce_FastMSS_recorded_room_reference_CPU_v1')
t=t.replace('fastmss_window_aggregate_feature_protocol_v1','fastmss_recorded_room_feature_protocol_v1')
t=t.replace("KIND = 'ASR18_actual_isolated_public_CPU_window_validation_v1'", "KIND = 'ASR18_actual_recorded_room_CPU_window_benchmark_v1'")
t=t.replace("PROPOSAL = 'ASR18_isolated_public_CPU_window_validation_proposal_v1'", "PROPOSAL = 'ASR18_recorded_room_aggregate_CPU_window_configuration_v1'")
t=change(t,'def main(proposal_path, output):','def main(proposal_path, ROOT_review, output):')
start=t.index('    core.require(proposal[\'kind\'] == PROPOSAL');end=t.index('    root = private_output(output)',start)
t=t[:start]+'''    review_pin = core.record(ROOT_review); review = core.read_bound(review_pin, core.resource_guard)
    core.require(proposal['kind'] == PROPOSAL and proposal['production_enabled'] is False
        and proposal['native_benchmark_only'] is True and review['accepted'] is True and review['reviewer'] == '/root'
        and core.same(review['configuration'], proposal_pin)
        and review['scope'] == 'recorded_noise_16room_aggregate_CPU_first4_benchmark_v1',
        'Exact reviewed production-disabled native-readable CPU benchmark configuration')
'''+t[end:]
t=change(t,"'numerical_request': entries[0], 'output_root': str(root)}",
    "'numerical_request': entries[0], 'output_root': str(root), 'ROOT_review': review_pin, 'configuration': proposal_pin}")
t=change(t,"core.same(value['prepared_payload'], original['prepared_payload'])",
    "value['prepared_payload']['state'] == 'prepared_not_committed'")
t=change(t,"for k in expected if k not in ('input_features', 'attention_mask')",
    "for k in expected if k not in ('input_features', 'attention_mask', 'prepared_PCM_sha256')")
t=change(t,"and actual['dtype'] == old['dtype'] and actual['array_sha256'] == old['array_sha256']",
    "and actual['dtype'] == old['dtype']")
t=change(t,"and actual['file']['sha256'] == old['file']['sha256'] and actual['file']['path'] != old['file']['path']",
    "and actual['file']['path'] != old['file']['path']")
t=change(t,"'All128 complete new NPY file bytes + raw arrays must exactly match the immutable old oracle')",
    "'All128 complete new arrays retain geometry/dtype and separate owned files; changed TR bytes are explicit')\n"
    "                if row['role'] == 'replay':\n"
    "                    core.require(actual['array_sha256'] == old['array_sha256']\n"
    "                        and actual['file']['sha256'] == old['file']['sha256']\n"
    "                        and row['prepared_PCM_sha256'] == expected['prepared_PCM_sha256'],\n"
    "                        'Replay/teacher clean PCM and all replay feature/mask bytes stay exact')")
t=change(t,"    result = {'kind': KIND",'''    operations = []
    applied_rows = clean_control_rows = 0
    for payload in payloads:
        for event in payload['rows']:
            aug = event['augmentation']
            if aug['applied']:
                applied_rows += 1
                operations.extend({'draw': event['draw'], **op} for op in aug['operations'])
            else: clean_control_rows += 1
    operation_counts = {kind: sum(v['kind'] == kind for v in operations) for kind in ('noise', 'rir')}
    result = {'kind': KIND''')
t=change(t,"'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,\n        'array_write",
    "'actual_optimizer_updates': 0, 'sampler_journal_frontier_advanced': False,\n"
    "        'all_children_actually_closed0': True, 'actual_CPU_aggregate_child_count': 2,\n"
    "        'native_benchmark_only': True, 'native_reader_source': core.record(public.__file__, guard),\n"
    "        'augmentation_identity': source_contract['augmentation_identity'], 'operation_counts': operation_counts,\n"
    "        'applied_rows': applied_rows, 'clean_control_or_protected_rows': clean_control_rows,\n"
    "        'actual_augmentation_operations': operations, 'no_Gaussian_or_new_asset_generation': True,\n"
    "        'array_write")
t=change(t,"'actual_isolated_public_CPU_protocol_window_completed': True, 'native_admission_claimed': False,",
    "'actual_recorded_room_CPU_window_completed': True, 'native_admission_claimed': False,")
start=t.index("if __name__ == '__main__':")
t=t[:start]+'''if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--ROOT-review', required=True); parser.add_argument('--folder', required=True)
    args = parser.parse_args(); main(args.config, args.ROOT_review, args.folder)
'''
source(controller,t)

area=Path(r'E:/gpu_training_scratchpad/analysis/FastMSS_real_recorded_noise_listening_v1')
policy={'kind':'FastMSS_recorded_noise_16room_TRAIN_policy_v1','benchmark_enabled':True,
    'production_enabled':False,'execution_requires_ROOT_review':True,'seed':20261017,'sample_rate':16000,
    'max_waveform_seconds':120,'apply_probability':.5,'noise_probability':.5,'rir_probability':.5,
    'snr_db_choices':[20,10,5],'noise_excerpt_seconds':30,'peak_ceiling':.98,
    'noise_mode':'verified_recorded_assets_only',
    'RIR_length_policy':'strongest_tap_unit_L2_full_convolution_keep_original_N_RMS_restore',
    'source_speech_cropped':False,'preview_full_tail_policy_adopted':False,
    'noise_registry':pin(area/'real_noise_registry_v1.json'),
    'room_registry':pin(area/'expanded_room_bank_v1/room_RIR_registry_v1.json')}
policy_path=F/'recorded_room_TRAIN_benchmark_policy_v1.json';write(policy_path,policy)
identity={'kind':'FastMSS_recorded_noise_16room_ordered_draw_identity_v1','policy_file':pin(policy_path),
    'policy_sha256':hashlib.sha256(json.dumps(policy,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
    'adapter':pin(adapter),'mixer':pin(S/'fastmss_real_noise_v1.py'),'decoder':pin(S/'fastmss_real_noise_train_adapter_v1.py'),
    'room_provenance_helper':pin(S/'fastmss_room_bank_preview_v1.py'),'noise_registry':policy['noise_registry'],
    'room_registry':policy['room_registry'],'draw_owner':'immutable original sampler source/epoch/draw_index',
    'commit_owner':'native caller after actual positive optimizer update','production_enabled':False,
    'new_draw_counter_or_global_RNG_advance':False}
contract=copy.deepcopy(load(F/'public_CPU_window_source_contract_v1.json'))
contract.update(kind='FastMSS_recorded_room_CPU_source_and_numerical_contract_v1',
    mode='recorded_room_first4_native_benchmark_only',validation_only=False,native_benchmark_only=True,
    augmentation_policy=pin(policy_path),augmentation_identity=identity)
bridge=load(contract['bridge']['path']);contract['original_augmentation_identity']=bridge['augmentation_identity']
legacy=load(contract['numerical_CPU8_contract']['path'])
contract['numerical_runtime_versions']={
    'NumPy':bridge['augmentation_identity']['numpy_version'],'SciPy':bridge['augmentation_identity']['scipy_version']}
contract['sources'].update(protocol=pin(protocol),reference=pin(reference),frontend=pin(frontend),controller=pin(controller),
    augmentation_adapter=pin(adapter),real_noise_mixer=identity['mixer'],real_noise_decoder=identity['decoder'],room_provenance=identity['room_provenance_helper'])
contract_path=F/'recorded_room_CPU_source_contract_v1.json';write(contract_path,contract)
old=load(F/'public_CPU_window_fixture_proposal_v1.json')
cfg=copy.deepcopy(old);cfg.update(kind='ASR18_recorded_room_aggregate_CPU_window_configuration_v1',
    source_contract=pin(contract_path),native_benchmark_only=True,production_enabled=False,
    output_root=str(F/'recorded_room_CPU_window_actual_01'),window_id='recorded_room_first4_01')
cfg['sources'].update(parent=pin(controller),reference=pin(reference),frontend=pin(frontend),protocol=pin(protocol),
    augmentation_adapter=pin(adapter),real_noise_mixer=identity['mixer'],real_noise_decoder=identity['decoder'],room_provenance=identity['room_provenance_helper'])
cfg_path=F/'recorded_room_CPU_configuration_v1.json';write(cfg_path,cfg)
manifest={'kind':'FastMSS_recorded_room_CPU_source_interface_manifest_v1','source_only':True,
    'sources':contract['sources'],'configuration':pin(cfg_path),'source_contract':pin(contract_path),
    'augmentation_policy':pin(policy_path),'production_enabled':False,'actual_execution_performed':False}
manifest_path=F/'recorded_room_CPU_source_interface_manifest_v1.json';write(manifest_path,manifest)
proposal={'kind':'FastMSS_recorded_room_CPU_first4_source_resource_proposal_v1','source_only':True,
    'configuration':pin(cfg_path),'source_contract':pin(contract_path),'source_manifest':pin(manifest_path),
    'scope':'recorded_noise_16room_aggregate_CPU_first4_benchmark_v1','sources':contract['sources'],
    'resource':contract['resource_review'],'raw_capacity_bytes':128*1024**2,'feature_capacity_bytes':512*1024**2,
    'new_raw_extension':{'maximum_N':120*16000,'stage_raw_peak_formula':'max(original,40*N+24MiB)',
        'maximum_raw_peak_bytes':40*120*16000+24*1024**2,'FFT_workspace_formula':'96*next_power_of_two(N+7*16000-1)',
        'maximum_FFT_workspace_bytes':96*2**21,'existing_transform_allowance_bytes':268435456,
        'numerical_runtime_reserve_bytes':805306368,'one_actual_row':True,'no_decoded_asset_cache':True,
        'compressed_domain_bytes':32*1024**2,'private1_8GB_preallocation_admission_unchanged':True},
    'fixed_scope':{'updates':4,'rows':64,'NPY_files':128,'actual_CPU_leaf_processes':2,
        'full_targets_and_original_PCM_retained':True,'old_SHARED_oracle_untouched':True,
        'native_consumption_requires_separate_ROOT_full_benchmark_review':True},
    'timing':{'whole_CPU_includes_parent_startup_source_asset_hash_decode_augmentation_frontend_write_fsync_readback_inventory':True,
        'whole_end_to_end_timer_owned_by_memory_native_harness':True,'cold_setup_and_per_row_phases_reported':True,
        'moving_hash_decode_work_outside_timer_for_speed_claim':False},
    'limits':['No actual source/runtime/audio/GPU job performed by this handoff.',
        'Original-N strongest-tap aligned training RIR differs from listening full tails, explicitly recorded.',
        'Replay/teacher/cleaner/DEV/TEST clean; >120s complete source retained clean; no draw/source exclusion.',
        'New native-readable BENCHMARK identity only, never relabel or adopt earlier validation-only output.',
        'New augmented TR bytes have no old-pilot equivalence claim. Clean source/targets/order and replay bytes remain exact.',
        'Unchanged CPU1.934217728GB tree; external combined supervisor CPUphase64M gives1.998217728GB total below2GB.',
        'First4 measured wall is not full population or20-window throughput; later scale/adoption separately reviewed.'],
    'production_enabled':False}
proposal_path=F/'recorded_room_CPU_source_resource_proposal_v1.json';write(proposal_path,proposal)
handoff={'kind':'FastMSS_recorded_room_CPU_source_handoff_v1','source_only':True,'production_enabled':False,
    'controller':pin(controller),'reader':pin(protocol),'configuration':pin(cfg_path),
    'source_contract':pin(contract_path),'policy':pin(policy_path),'manifest':pin(manifest_path),'proposal':pin(proposal_path),
    'CLI':['--config',str(cfg_path),'--ROOT-review','EXTERNAL_ROOT_REVIEW_PATH','--folder',cfg['output_root']]}
handoff_path=F/'recorded_room_CPU_source_handoff_v1.json';write(handoff_path,handoff)
print(json.dumps({'handoff':pin(handoff_path),'controller':pin(controller),'reader':pin(protocol),
    'adapter':pin(adapter),'config':pin(cfg_path),'contract':pin(contract_path),'policy':pin(policy_path),
    'manifest':pin(manifest_path),'proposal':pin(proposal_path)},indent=2))
