"""Metadata-only two-field configuration and exact existing saved-state gate."""
import argparse,ast,copy,hashlib,json,sys
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=P/'work/research';S=P/'outputs/parakeet_turkish';sys.path[:0]=[str(S),str(R)]
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from run_ASR18_BPE9223_resident_RAM_training_v9 import checked_original_interval_resume
p=argparse.ArgumentParser();p.add_argument('--CPU-review',required=True);a=p.parse_args();N=Path(__file__).parent;OUT=N/'production_configuration_v1';require(not OUT.exists(),'Fresh exact metadata output');OUT.mkdir()
review_pin=record(a.CPU_review);review=read_json(a.CPU_review)
require(review['accepted'] is True and review['reviewer']=='/root' and review['scope']=='ASR18_source_bound_second_original_full_clip_geometry_overlay_v1'
 and review['overlay_source']==record(S/'fastmss_original_clip_geometry_overlay_v2.py') and review['certificate']==record(N/'second_original_full_clip_geometry_certificate_v1.json'),'Actual reviewed overlay/certificate')
source_config=N.parent/'resident_RAM_original_interval32406_failure_actual_v1/production_configuration_v1/configuration.json';base=read_json(source_config);config=copy.deepcopy(base)
config['original_clip_geometry_overlay']={'prior_overlay':copy.deepcopy(base['original_clip_geometry_overlay']),'source':record(S/'fastmss_original_clip_geometry_overlay_v2.py'),'certificate':record(N/'second_original_full_clip_geometry_certificate_v1.json'),'ROOT_review':review_pin}
config['prior_resident_RAM_original_interval_failure']=record(N/'actual_failure_closure.json')
path=OUT/'configuration.json';write_json(path,config)
check=copy.deepcopy(config);check.pop('original_clip_geometry_overlay');check.pop('prior_resident_RAM_original_interval_failure');old_remaining=copy.deepcopy(base);old_remaining.pop('original_clip_geometry_overlay');old_remaining.pop('prior_resident_RAM_original_interval_failure');require(check==old_remaining,'Exactly two existing operational fields changed; identity/science/resources unchanged')
failure=read_json(N/'actual_failure_closure.json');pointer=failure['checkpoint_pointer'];branch=Path(failure['branch'])
value=checked_original_interval_resume(config['prior_resident_RAM_original_interval_failure'],base=config,folder=branch,pointer_pin=pointer)
from fastmss_original_clip_geometry_overlay_v2 import checked_certificate
certificate=checked_certificate(config['original_clip_geometry_overlay'],lambda:None)
require(certificate['actual_failure']==config['prior_resident_RAM_original_interval_failure'],'One exact failure/correction lineage')
def defs(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(source).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
old=defs(R/'run_ASR18_BPE9223_resident_RAM_training_v8.py');new=defs(R/'run_ASR18_BPE9223_resident_RAM_training_v9.py');same=[k for k in old if old[k]==new[k]]
require(all(k in ('main','checked_original_interval_resume') or k in same for k in old),'All existing non-entry functions/classes including native transaction unchanged')
prior=read_json(N.parent/'ROOT_original_full_clip_saved32400_execution_review_v1.json')
h={'kind':'ASR18_exact_saved45200_original_full_clip_continuation_execution_handoff_v1','configuration':record(path),'prior_configuration':record(source_config),'only_changed_configuration_fields':['original_clip_geometry_overlay','prior_resident_RAM_original_interval_failure'],'driver_source':record(R/'run_ASR18_BPE9223_resident_RAM_training_v9.py'),'wrapper_source':record(R/'run_ASR18_BPE9223_resident_RAM_autocontinue_v9.py'),'CPU_source':record(S/'fastmss_resident_feature_stream_v3.py'),'RAM_native_source':prior['RAM_native_source'],'DEV_helper_source':prior['DEV_helper_source'],'original_clip_geometry_overlay':config['original_clip_geometry_overlay'],'actual_CPU_closure':record(N/'actual_ingress_frontend_CPU_closure_v1.json'),'actual_CPU_report':record(N/'ingress_frontend_actual_01/report.json'),'original_interval_resume':config['prior_resident_RAM_original_interval_failure'],'initial_checkpoint':pointer,'checkpoint_persistence':failure['checkpoint_persistence'],'latest_failure':prior['latest_failure'],'unchanged_non_entry_driver_functions':same,'actual_serialized_failure_and_certificate_checkers_passed':True,'whole_original_journal_preserved':value['whole_journal_including_preserved_tail'],'preserved_tail_positive_updates':value['preserved_tail_positive_updates'],'new_journal_path':str(branch/'journals/commits_000008.jsonl'),'bounded_RAM_queue_bytes':268435456,'periodic_DEV_interval_updates':4000,'start_invocation':8,'full_target':401072,'administrative_update_stop':None,'branch':str(branch),'output':str(N.parent/'resident_RAM_training_actual_09'),'private_scalar_prefetch_source':prior['private_scalar_prefetch_source'],'private_scalar_prefetch_actual_CPU_review':prior['private_scalar_prefetch_actual_CPU_review'],'additional_prefetch_host_reservation_bytes':268435456,'execution_authorized':False}
write_json(OUT/'execution_handoff.json',h);print(json.dumps({'handoff':record(OUT/'execution_handoff.json'),'configuration':record(path),'serialized_checkers_passed':True}),flush=True)
