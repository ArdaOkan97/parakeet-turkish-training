"""Correct inherited configuration prose/CLI only; all frozen sources preserved."""
import copy,hashlib,json
from pathlib import Path
F=Path(r'E:/gpu_training_scratchpad/analysis/FastMSS_integration_v1')
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20')
def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    path=Path(path).resolve();raw=path.read_bytes()
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
cfg=copy.deepcopy(load(F/'recorded_room_CPU_configuration_v1.json'))
cfg_path=F/'recorded_room_CPU_configuration_v2.json'
cfg['execution']={
    'actual_execution_authorized':False,'external_ROOT_configuration_review_required':True,
    'script':cfg['sources']['parent'],
    'arguments':['--config',str(cfg_path),'--ROOT-review','EXTERNAL_ROOT_CPU_REVIEW_PATH','--folder',cfg['output_root']],
    'guard_source':cfg['sources']['guard'],
    'outer_guard_folder':str(F/'recorded_room_CPU_window_guard_actual_01'),
    'owned_log':str(F/'recorded_room_CPU_window_owned_01.log'),
    'coordinate_with_memory_owner':True}
cfg['namespace'].update(path=r'E:\gpu_training_scratchpad\temp\ASR18_recorded_room_CPU_benchmark_v1',
    role='recorded_room_first4_native_benchmark_only',production_source_eligible=False,native_benchmark_only=True)
cfg['namespace']['separate_allocated_benchmark_feature_reservation_bytes']=cfg['namespace'].pop('separate_allocated_validation_feature_reservation_bytes')
cfg['scope']=[
    'Same original first4 updates/64 immutable ordered draws and complete targets, independent clean whole-PCM references.',
    'Two genuine aggregate CPU leaves. One reference actual0 precedes one recorded-noise/16room/frontend actual0, never fake per-update exits.',
    'NEW explicit source-bound training augmentation overlay; old Gaussian/twoRIR policy/session remains immutable historical planning input.',
    'All128 new NPY file/raw hashes, dtype/shape,64 contexts/targets and4 NEW policy-bound prepared payloads recorded; changed TRAIN bytes are intentional, no old-pilot equivalence claim.',
    'Independent clean-reference/whole-original-PCM/targets/order equal old baseline; replay/teacher PCM and replay feature/mask bytes exact.',
    'TRAIN/TR only, <=120s; original-N strongest-tap aligned RIR then recorded noise at whole-clip RMS SNR. Full original input retained; preview tails not used.',
    'Fresh native-readable BENCHMARK namespace, production/source eligibility false; no earlier validation output adoption or original shared oracle mutation.',
    'Full write/fsync/hash/readback/inventory included in measured CPU wall; asset/source validation/decode/setup and per-row timings included.',
    'Single quiescent frontend collect/freeze; all original per-row GC/weakref/raw/frontend/heldout checks retained, unfreeze before teardown.',
    'Actual native consumption/checkpoint/recovery must be separately source/config admitted in the combined memory-owner harness. No CPU-only output implies native or training approval.']
cfg['remaining_gates']=[
    'ROOT exact new adapter/raw-extension/protocol/worker/controller/config/resource review before one actual first4 CPU job.',
    'Actual2 aggregate0+parent/worker clean guards/all family dead, complete128 files/full finite arrays/new policy-bound payloads.',
    'Separately reviewed native harness must consume this exact owner/manifest using new reader and preserve its ordered uncommitted payload in actual checkpoint/journal state.',
    'No automatic cleanup/deletion; failed and successful fresh benchmark namespaces preserved for separately reviewed owned retirement.',
    'Whole CPU wall and whole CPU-to-native/checkpoint/recovery wall are separate from native update timing; no full-run or20-window scaling claim.']
cfg['source_only_validation'].update(AST_parse_five_new_sources=True)
cfg['source_only_validation'].pop('AST_parse_four_new_sources',None)
write(cfg_path,cfg)
m=copy.deepcopy(load(F/'recorded_room_CPU_source_interface_manifest_v1.json'))
m.update(kind='FastMSS_recorded_room_CPU_source_interface_manifest_v2',configuration=pin(cfg_path),
    sole_successor_delta='Configuration metadata CLI/namespace/expectation/prose correction only. All five source files and policy/contract unchanged.')
mp=F/'recorded_room_CPU_source_interface_manifest_v2.json';write(mp,m)
proposal=copy.deepcopy(load(F/'recorded_room_CPU_source_resource_proposal_v1.json'))
proposal.update(kind='FastMSS_recorded_room_CPU_first4_source_resource_proposal_v2',configuration=pin(cfg_path),source_manifest=pin(mp),
    sole_successor_delta=m['sole_successor_delta'])
pp=F/'recorded_room_CPU_source_resource_proposal_v2.json';write(pp,proposal)
h=copy.deepcopy(load(F/'recorded_room_CPU_source_handoff_v1.json'))
h.update(kind='FastMSS_recorded_room_CPU_source_handoff_v2',configuration=pin(cfg_path),manifest=pin(mp),proposal=pin(pp),
    CLI=cfg['execution']['arguments'])
hp=F/'recorded_room_CPU_source_handoff_v2.json';write(hp,h)
print(json.dumps({'handoff':pin(hp),'config':pin(cfg_path),'manifest':pin(mp),'proposal':pin(pp)},indent=2))
