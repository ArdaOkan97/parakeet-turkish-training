"""Inactive certificate/source serialization only; no candidate import or proof."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import psutil
P=Path('C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish'
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_source36_original_interval186563_failure_actual_v1')
def require(v,m):
    if not v:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def pin(p):
    p=Path(p).resolve();data=p.read_bytes();require(len(data)<=8388608,'Compact/source only')
    return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def encode(v):return (json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
def future(p,b):return {'path':str(p.resolve()),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def identity(p):return {'pid':p.pid,'create_time':p.create_time(),'ppid':p.ppid()}
me=psutil.Process();me.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS);sys.path.insert(0,str(S))
from resource_watchdog_v2 import sample_memory
def guard():
    v=sample_memory(me.pid,'inactive_ninth_certificate_overlay10_CPU14_SOURCE_preparation')
    require(v['paired_sampling_within_one_second']and v['worker_private_memory']['private_bytes']<=536870912
        and v['system_memory']['commit_available_bytes']>=5055306368 and v['system_memory']['physical_available_bytes']>=2305306368
        and shutil.disk_usage('E:/').free>=100000000000,'Metadata full prospective/floors/FULLprefetch/E100GB')
    return v
initial=guard();scope_path=F/'ROOT_recorded186564_row5_NINTH_certificate_overlay10_CPU14_INACTIVE_preparation_scope_v1.json';scope_pin=pin(scope_path)
require(scope_pin['bytes']==27371 and scope_pin['sha256']=='fa409fdaeb7a91158377c73b712dc2306f6353040b85a951a13616e943e1db95','Exact preparation-only scope')
scope=read(scope_path);root_pin=scope['actual_physical_ROOT'];require(pin(root_pin['path'])==root_pin,'Accepted ROOT physical receipt exact')
root=read(root_pin['path']);where=scope['exact_original_row_identity'];physical=scope['physical_binding'];geometry=scope['actual_derived_geometry']
require(root['accepted']is True and root['reviewer']=='/root'and root['physical']==physical
    and root['independent_arithmetic']['derived_geometry']==geometry
    and physical['complete_demux_decode_EOF']is True and physical['all_original_physical_samples_used_once']is True
    and physical['synthesized_or_skipped_samples']==0,'Use accepted ROOT scalar physical binding only, no raw observation reads')
raw=where['raw_original_annotation'];item=where['original_plan_row']
certificate={'kind':'ASR18_source_bound_one_original_full_clip_geometry_certificate_v1',
    'original_row':deepcopy(raw),'original_row_sha256':digest(raw),'original_declaration':deepcopy(scope['original_larger_capacities']),
    'full_target_ids':deepcopy(where['full_target_ids']),'full_target_ids_sha256':digest(where['full_target_ids']),
    'raw_label_sha256':hashlib.sha256(raw['text'].encode('utf-8')).hexdigest(),
    'original_TAR':{'path':scope['source_artifact']['path'],'bytes':scope['source_artifact']['bytes'],
        'mtime_ns':scope['source_artifact']['mtime_ns'],'recorded_sha256':scope['source_artifact']['sha256']},
    'recording_index_entry':dict(zip(('member_name','offset','bytes'),scope['recording_index_entry'])),
    'member_sha256':physical['compressed_member_sha256'],
    'actual_geometry':{k:geometry[k]for k in('native_sr','native_samples','clean16k_samples','mel_frames','encoder_frames','target_tokens')},
    'decoded_frames':physical['decoded_frames'],'max_abs_PTS_residual_samples':physical['max_abs_PTS_residual_samples'],
    'native_f32_sha256':physical['native_f32_sha256'],'clean_f32_sha256':physical['clean_f32_sha256'],'clean_i16_sha256':physical['clean_i16_sha256'],
    'independent_physical_ROOT_review':root_pin,'ordered_failure_localization_ROOT':scope['existing_localization_ROOT'],
    'actual_failure_ROOT_review':pin(F/'ROOT_source36_failure186563_saved186400_actual_independent_verification_v1.json'),
    'original_annotations_indexes_plans_full66_and_T165_unchanged':True,'all_original_physical_samples_once':True,
    'no_generic_EOF_tolerance_or_clamp':True,'historical_native_reservation_retained':630864,
    'actual_physical_native_samples':630840,'historical_clean_reservation_retained':210288,'actual_full_clean_samples':210280,
    'unchanged_mask1314_features1315_T165_all66_targets':True,'actual_raw_codec_side_data':deepcopy(physical['side_data']),
    'full_proof_prospective901306112_retained_no_freed_credit':True,'side_discard528_not_a_sole_cause_proof':True}
cert_path=F/'ninth_original_full_clip_geometry_certificate_v1.json';cert_bytes=encode(certificate)
old_overlay=S/'fastmss_original_clip_geometry_overlay_v9.py';new_overlay=S/'fastmss_original_clip_geometry_overlay_v10.py'
old_cpu=S/'fastmss_resident_feature_stream_v13.py';new_cpu=S/'fastmss_resident_feature_stream_v14.py'
old_text=old_overlay.read_bytes().decode('utf-8');text=old_text;edits=[]
def change(before,after,count=1):
    global text
    require(text.count(before)==count,'Exact replacement count: '+before[:90]);text=text.replace(before,after)
    edits.append({'old_literal':before,'new_literal':after,'expected_count':count})
change('Eighth exact source-bound EOF certificate','Ninth exact source-bound EOF certificate')
change('physical samples and explicitly reports mel1710 versus declared1710, with T214/U99.',
    'physical samples and explicitly reports mel1314 versus declared1314, with T165/U66.')
change('import fastmss_original_clip_geometry_overlay_v7 as prior','import fastmss_original_clip_geometry_overlay_v9 as prior')
change('from fastmss_original_clip_geometry_overlay_v7 import','from fastmss_original_clip_geometry_overlay_v9 import')
change("ROW_ID='alania:aedf14e56cf50416170171503e0dee9a_00000'","ROW_ID='alania:a280321c852958fbb53e9d7a2044d1a6_00000'")
change('ORDINAL=332560','ORDINAL=156134')
change('ASR18_source_bound_eighth_original_full_clip_geometry_overlay_v1','ASR18_source_bound_ninth_original_full_clip_geometry_overlay_v1')
change("and certificate['original_row']['end'] == 17.102","and certificate['original_row']['end'] == 13.143")
change("and len(certificate['full_target_ids']) == 99","and len(certificate['full_target_ids']) == 66")
change('native_samples=820896, clean16k_samples=273632, mel_frames=1710,\r\n            encoder_frames=214, target_tokens=99',
    'native_samples=630864, clean16k_samples=210288, mel_frames=1314,\r\n            encoder_frames=165, target_tokens=66')
change('native_samples=820872,\r\n            clean16k_samples=273624, mel_frames=1710, encoder_frames=214, target_tokens=99',
    'native_samples=630840,\r\n            clean16k_samples=210280, mel_frames=1314, encoder_frames=165, target_tokens=66')
change("and certificate['decoded_frames'] == 856","and certificate['decoded_frames'] == 658")
before=old_text[old_text.index("    observed = core.read_bound(certificate['independent_full_member_observation'], guard)"):old_text.index('    return certificate')]
after="""    observed = core.read_bound(certificate['independent_physical_ROOT_review'], guard)
    require(observed['accepted'] is True and observed['reviewer'] == '/root'
            and observed['kind'] == 'ASR18_recorded186564_row5_full_original_member_physical_actual01_independent_ROOT_verification_v1',
            'Exact independently accepted ROOT physical receipt, no closed raw observation re-audit')
    physical = observed['physical']; identity = observed['exact_original_row_identity']
    source = observed['source_artifact']; entry = observed['recording_index_entry']
    require(physical['complete_demux_decode_EOF'] is True
            and physical['all_original_physical_samples_used_once'] is True
            and physical['synthesized_or_skipped_samples'] == 0
            and identity['raw_original_annotation'] == certificate['original_row']
            and identity['full_target_ids'] == certificate['full_target_ids']
            and {k:observed['independent_arithmetic']['derived_geometry'][k] for k in actual} == actual
            and physical['native_f32_sha256'] == certificate['native_f32_sha256']
            and physical['clean_f32_sha256'] == certificate['clean_f32_sha256']
            and physical['clean_i16_sha256'] == certificate['clean_i16_sha256']
            and physical['compressed_member_sha256'] == certificate['member_sha256']
            and dict(zip(('member_name', 'offset', 'bytes'), entry)) == certificate['recording_index_entry']
            and dict(path=source['path'], bytes=source['bytes'], mtime_ns=source['mtime_ns'],
                     recorded_sha256=source['sha256']) == certificate['original_TAR']
            and observed['resource']['original_prospective'] == 901306112,
            'Exact ROOT physical EOF/full labels/source/member/hashes and larger original reservations')
""".replace('\n','\r\n')
change(before,after)
change('self.eighth','self.ninth',old_text.count('self.eighth'))
change('seven prior certified rows','eight prior certified rows')
new_bytes=text.encode('utf-8');ast.parse(new_bytes)
inverse=text
for edit in reversed(edits):
    require(inverse.count(edit['new_literal'])==edit['expected_count'],'Exact reverse literal count')
    inverse=inverse.replace(edit['new_literal'],edit['old_literal'])
require(inverse.encode('utf-8')==old_overlay.read_bytes(),'Full prior overlay9 RAW byte inverse')
cpu_old=old_cpu.read_bytes();old_import=b'from fastmss_original_clip_geometry_overlay_v9 import OriginalRows';new_import=b'from fastmss_original_clip_geometry_overlay_v10 import OriginalRows'
require(cpu_old.count(old_import)==1,'Exactly ONE CPU13 overlay import');cpu_bytes=cpu_old.replace(old_import,new_import)
require(cpu_bytes.replace(new_import,old_import)==cpu_old,'Full CPU13 RAW inverse after ONLY one import')
ast.parse(cpu_bytes)
# Static exported alias resolution, without importing any overlay/CPU.
alias_chain={}
def resolve_alias(path,symbol,seen=()):
    require((str(path),symbol)not in seen,'No circular exported alias')
    tree=ast.parse(path.read_bytes());steps=[{'module':pin(path),'symbol':symbol}]
    if any(isinstance(n,ast.FunctionDef)and n.name==symbol for n in tree.body):return steps
    assigned=next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id==symbol for t in n.targets))
    names=[];value=assigned
    while isinstance(value,ast.Attribute):names.append(value.attr);value=value.value
    require(isinstance(value,ast.Name),'Explicit static module-alias export')
    names=[value.id,*reversed(names)];target=path
    for alias in names[:-1]:
        module=ast.parse(target.read_bytes())
        imported=next(n.names[0].name for n in module.body if isinstance(n,ast.Import)and n.names[0].asname==alias)
        target=S/(imported+'.py')
    return steps+resolve_alias(target,names[-1],seen+((str(path),symbol),))
for symbol in('match_original','decode_full_member'):
    chain=resolve_alias(old_overlay,symbol)
    require(Path(chain[-1]['module']['path']).name=='fastmss_original_clip_geometry_overlay_v1.py'
        and chain[-1]['symbol']==symbol,'Both static exported aliases end at accepted original1 definition')
    alias_chain[symbol]=chain
ledger_path=F/'ninth_overlay10_CPU14_full_RAW_inverse_and_aliases_v1.json'
ledger={'kind':'ASR18_INACTIVE_ninth_overlay10_CPU14_full_RAW_inverse_v1','newline_normalization':'NONE, exact original CRLF bytes preserved',
    'overlay':{'old':pin(old_overlay),'new':future(new_overlay,new_bytes),'replacements':edits,'entire_old_RAW_reconstructed':True},
    'CPU':{'old':pin(old_cpu),'new':future(new_cpu,cpu_bytes),'replacements':[{'old_literal':old_import.decode(),'new_literal':new_import.decode(),'expected_count':1}],
        'entire_old_RAW_reconstructed':True,'only_ONE_import_change':True},
    'static_export_chain_for_both_match_original_and_decode_full_member':alias_chain,
    'no_alias_import_or_execution':True,'all_prior8_sources_certificates_configuration_tree_unchanged':True}
ledger_bytes=encode(ledger)
handoff_path=F/'ninth_overlay10_CPU14_INACTIVE_source_gate_handoff_v1.json'
handoff={'kind':'ASR18_INACTIVE_NINTH_overlay10_CPU14_source_gate_handoff_v1','execution_admitted':False,
    'preparation_ROOT_scope':scope_pin,'actual_physical_ROOT':root_pin,'certificate':future(cert_path,cert_bytes),
    'overlay_source':future(new_overlay,new_bytes),'CPU14_source':future(new_cpu,cpu_bytes),'proper_CPU13_source':pin(old_cpu),
    'prior_overlay9_source':pin(old_overlay),'full_RAW_inverse':future(ledger_path,ledger_bytes),
    'required_overlay_constructor_review_scope':'ASR18_source_bound_ninth_original_full_clip_geometry_overlay_v1',
    'required_overlay_constructor_review_fields':['accepted:true','reviewer:/root','scope','overlay_source','certificate'],
    'required_genuine_derived_CPU14_transport_successor':'Original accepted323c transport proof reused through exact CPU13->14 import-only inverse; no new transport test or proof weakening',
    'CPU14_transport_review_PENDING_no_fake_completed_review':True,
    'next_cpu_ingress_fixture_configuration_isolated_only':'Will be constructed inside future admitted proof output from current I25/f866; no production driver/wrapper/config materialization now',
    'original_full66_T165_mask1314_features1315_and_larger630864_210288_reservations':True,
    'CPU2_5GB_private8GB_FULLprefetch_full901306112_envelope_225_MAX2_LIVE1_unchanged':True,
    'no_frontend_proof_fit_speed_prevention_recovery_or_production_admission':True,
    'typed_transition_SOURCE_ONLY_proposal':{'proper_base':'current source36 + actual I25/f866, not old source33',
        'future_actual_failure_binding':'accepted actual186563 closure/current saved186400 +163 historical redo, new invocation26/journal26/outputA27 only after final review',
        'first_idle_event':'Original process consumed its first180k periodic event. Future continuation must preserve consumed status and MUST NOT schedule a second refresh; explicit inherited consumed receipt needs exact typed proof.',
        'required_metadata_fixture':'Entire new driver pre/postrestore factory/branch dict and wrapper require+forwarding, full actual failure/restore source/current7 binding kinds and both config values; old lanes inverse exact',
        'recovery_then':'Full854 allstate saved186400 restore-before-prefixcopy, genuine163 redo+new186564 repaired positive, natural186600/post186620, reader unchanged',
        'production_source_or_configuration_materialization_NOT_admitted':True}}
prepared={cert_path:cert_bytes,new_overlay:new_bytes,new_cpu:cpu_bytes,ledger_path:ledger_bytes,handoff_path:encode(handoff)}
result_path=F/'ninth_overlay10_CPU14_source_preparation_result_v1.json'
require(all(not path.exists()for path in[*prepared,result_path]),'Fresh inactive source destinations BEFORE writes')
final=guard()
for path,data in prepared.items():path.write_bytes(data)
result={'kind':'ASR18_INACTIVE_ninth_overlay10_CPU14_preparation_result_v1','prepared':[pin(path)for path in prepared],
    'producer':identity(me),'launcher':identity(me.parent()),'preparation_source':pin(__file__),
    'initial_metadata_guard':initial,'final_metadata_guard':final,'no_candidate_import_proof_decode_frontend_or_production_action':True}
result_path.write_bytes(encode(result));print(json.dumps({'result':pin(result_path),**result}),flush=True)
