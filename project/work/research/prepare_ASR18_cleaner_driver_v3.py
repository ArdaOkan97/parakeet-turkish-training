"""New driver-only truthfulness delta; preserve frozen probe/config/bundlev2."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=Path(__file__).resolve().parent;sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    source=R/'run_ASR18_cleaner_hook_v2.py';target=source.with_name('run_ASR18_cleaner_hook_v3.py');old=source.read_text()
    before="passed and request['name'].startswith('native_integrated_update_')"
    after="passed and (request['name'].startswith('native_integrated_update_') or request['name']=='prospective_clean_native_update21')"
    require(old.count(before)==1 and not target.exists(),'Fresh single driver delta')
    target.write_text(old.replace(before,after),encoding='utf-8',newline='\n')
    require(target.read_bytes().replace(after.encode(),before.encode())==source.read_bytes(),'Raw byte inverse exactly reconstructs frozen driver2')
    delta={'kind':'ASR18_clean_hook_driver_optimizer_stage_truthfulness_delta_v3','old':record(source),'new':record(target),
        'single_boolean_stage_recognition_change':True,'raw_byte_inverse_restores_driver2':True,'old_expression':before,'new_expression':after,
        'source_probe_config_native_forecast_and_guards_unchanged':True,'GPU_launched':False}
    dp=A/'cleaner_hook_driver_stage_delta_v3.json';write_json(dp,delta)
    old_manifest=A/'production_cleaner_forecast_coordinator_review_manifest_v2.json';value=json.loads(old_manifest.read_text())
    require(all(record(v['path'])==v for v in value['bindings']),'Frozen v2 binding changed')
    value.update(kind='ASR18_frozen_clean1_prospective_forecast_coordinator_source_request_v3',
        prior_frozen_request=record(old_manifest),runtime_driver=record(target),driver_truthfulness_delta=record(dp),
        probe_and_config_version_unchanged_from_v2=True)
    value['sources'].append(record(target));value['bindings'].extend([record(target),record(dp),record(old_manifest)]);value['owned_transitive_python_sources']+=1
    path=A/'production_cleaner_forecast_coordinator_review_manifest_v3.json';require(not path.exists(),'Fresh immutable request');write_json(path,value)
    print(json.dumps({'manifest':record(path),'driver':record(target),'delta':record(dp),'GPU_launched':False}))
if __name__=='__main__':main()
