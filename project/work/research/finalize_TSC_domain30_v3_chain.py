"""Bind executed focused dispatch evidence to the NEW domain-preserving TSC chain."""
import argparse,ast,hashlib,json,re
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def replace(path,name,value):
    text=path.read_text(encoding='utf-8');text,n=re.subn(r'^'+re.escape(name)+r'\s*=.*$',name+' = '+repr(value),text,flags=re.M)
    assert n==1,(path,name,n);ast.parse(text);path.write_text(text,encoding='utf-8',newline='\n')

def main(args):
    oldpath=HERE/'TSC_runtime_chain_pins_v2.json'
    assert sha(oldpath)=='7f50e684b62bd250d4c3fe51248d44380a27aeeb8e6c627bceb47de6d2fde37b'
    old=json.loads(oldpath.read_text(encoding='utf-8'))
    for name,digest in old['source_sha256'].items():assert sha(HERE/name)==digest,name
    proofpath=args.dispatch_proof.resolve();proof=json.loads(proofpath.read_text(encoding='utf-8'))
    assert proof['kind']=='isolated_TSC_runtime_mocked_CPU_dispatch_verification_v3' and proof['passed'] is True
    assert proof['actual_model_or_GPU_execution'] is False and proof['exact_admission_checked'] is True
    replacements={stem+'_v2.py' for stem in ('cuda_cache_caller_tsc','launch_with_io_retry_tsc',
        'launch_with_cache_reclaim_tsc','launch_with_memory_guard_tsc','launch_native_training_tsc','train_lora_batched_tsc')}
    pins={name:digest for name,digest in old['source_sha256'].items() if name not in replacements}
    pins.update(proof['source_sha256'])
    assert len(pins)>=26 and len(pins)==len(set(pins))
    for name,digest in pins.items():assert sha(HERE/name)==digest,name
    handoff=args.worker_handoff.resolve();science=json.loads(handoff.read_text(encoding='utf-8'))
    assert handoff.is_relative_to(ROOT) and science['kind']=='TSC_domain30_scientific_worker_handoff_v3'
    assert science['passed_CPU_checks'] is True and science['actual_TSC_model_GPU_adoption'] is False
    science_proof=Path(science['CPU_proof']['path'])
    assert sha(science_proof)==science['CPU_proof']['sha256']
    assert json.loads(science_proof.read_text(encoding='utf-8'))['passed'] is True
    memory=HERE/'launch_with_memory_guard_TSC_resume_v3.py';assert proof['exact_memory_launcher']==record(memory)
    policy=dict(old['policy']);policy['sampling_mode']='domain30';policy['fleurs_fraction_of_TR_clips']=.3
    policy['replay_RNG_independent_of_Turkish']=True
    policy['declared_scientific_changes']=[
        'Add approved quarantined TSC only to70%otherTR domain; retain30%FLEURS and independent replay',
        'Original74883TR prefix,1595replay/teacher/targets and frozen8199/native3 model recipe unchanged',
        'InitialfreshAdam/local sampler from latest ROOT-chosen CLOSED qualifying original domain BEST',
        'epochs1/steps exact first integer quota with minimum FLEURS/CV+TSC exposure>=1; DEV250 feedback is not an epoch',
        'Later scheduled budget windows preserve fullSAME-run Adam/sampler/RNG/BEST/history and original native-ready']
    destination=HERE/'TSC_runtime_chain_pins_v3.json';assert not destination.exists()
    value={**old,'kind':'isolated_TSC_runtime_chain_pins_v3','status':'CPU_routing_prepared',
        'source_sha256':pins,'source_paths':{name:str(HERE/name) for name in pins},
        'worker_sha256':pins['train_lora_batched_tsc_v3.py'],
        'outer_launcher':record(HERE/'launch_native_training_tsc_v3.py'),
        'memory_launcher':record(HERE/'launch_with_memory_guard_tsc_v3.py'),
        'exact_memory_launcher':record(memory),'exact_memory_kind':'memory_guard_training_launcher_TSC_resume_v3',
        'exact_runtime_scope':'Separateprocess admission; originalv3 science/native-ready identity stays immutable',
        'caller_target':{'file':'train_lora_batched_tsc_v3.py','function':'run','line':proof['caller_line'],
            'inherited_operations':'Unchanged cuda_cache_reclaim_v2.py; new caller/source binding only'},
        'policy':policy,'source_closure_required_kind':'root_owned_TSC_domain30_closed_source_v3',
        'memory_identity_additional_fields':[*old['memory_identity_additional_fields'],'TSC_domain_sampler_sha256','TSC_domain_identity_sha256'],
        'CPU_dispatch_proof_path':str(proofpath),'CPU_dispatch_proof_sha256':sha(proofpath),
        'repeatable_fixture_path':proof['fixture_script_path'],'repeatable_fixture_sha256':proof['fixture_script_sha256'],
        'worker_scientific_handoff_path':str(handoff),'worker_scientific_handoff_sha256':sha(handoff),
        'prior_v2_pin_receipt':record(oldpath),'actual_TSC_import_or_training_approval_validated':False,
        'created_at_utc':datetime.now(timezone.utc).isoformat()}
    with destination.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)
    first=HERE/'run_hf_TSC_arm_v3.py';replace(first,'CHAIN_SHA',sha(destination))
    preload=HERE/'launch_TSC_same_science_resume_v3.py';replace(preload,'PINS',pins);replace(preload,'RUNTIME_MEMORY_SHA',sha(memory))
    exact=HERE/'resume_TSC_same_science_v3.py';replace(exact,'TSC_CONTROLLER_SHA',sha(first))
    supervisor=HERE/'supervise_TSC_same_science_v3.py'
    for name,path in (('FIRST_SHA',first),('HELPER_SHA',exact),('PRELOADER_SHA',preload)):replace(supervisor,name,sha(path))
    print(json.dumps({'receipt':record(destination),'source_count':len(pins),
        'orchestration':{p.name:sha(p) for p in (first,exact,preload,supervisor)},'exact_memory':record(memory)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dispatch-proof',type=Path,required=True);p.add_argument('--worker-handoff',type=Path,required=True);main(p.parse_args())
