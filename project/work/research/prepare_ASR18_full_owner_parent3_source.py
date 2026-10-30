"""Source-only standalone DEV supervision successor; no target imports/jobs."""
from pathlib import Path
import ast, hashlib

P=Path(__file__).resolve().parents[2]
old=P/'work/research/run_ASR18_full_run_owner_v2.py'
new=old.with_name('run_ASR18_full_run_owner_v3.py')
assert not new.exists()
text=old.read_text(encoding='utf-8-sig')
def replace(before,after):
    global text
    assert text.count(before)==1, before[:100]
    text=text.replace(before,after)

replace("self.completion_path=folder/('actual_current_recovery/current_recovery_report.json' if self.config['worker_mode']=='restore_only' else 'worker_completion.json')",
"self.completion_path=folder/('serial350_worker_report.json' if self.config['worker_mode']=='serial350DEV_optimizer_free' else\n            'actual_current_recovery/current_recovery_report.json' if self.config['worker_mode']=='restore_only' else 'worker_completion.json')")
replace("require(self.config['worker_mode'] in ('resident_training', 'restore_only'), 'Exact separated GPU worker mode')",
"require(self.config['worker_mode'] in ('resident_training', 'restore_only','serial350DEV_optimizer_free'), 'Exact separated GPU worker mode')")
replace("        else:\n            from asr18_full_run_current_recovery_v2 import checked_request",
"        elif self.config['worker_mode']=='serial350DEV_optimizer_free':\n            from asr18_full_run_DEV_runtime_v4 import resource_proposal,checked_configuration\n            checked_configuration(self.config,parent_source=record(__file__))\n            require(self.config['worker_driver_source']==record(SOURCE/'asr18_full_run_DEV_runtime_v4.py'),\n                    'DEV launches the actual standalone runtime4 entrypoint')\n            expected={'entire_worker_private_forecast_bytes':resource_proposal()['whole_worker_private_forecast_bytes']}\n        else:\n            from asr18_full_run_current_recovery_v2 import checked_request")
replace("        print(json.dumps({'actual_GPU_native_PID': self.pid, 'launcher_PID': self.child.pid, 'folder': str(folder)}), flush=True)",
"        if self.config['worker_mode']=='serial350DEV_optimizer_free':\n            require(Path(self.config['owned_worker_admission_path']).resolve()==folder/'owned_worker_admission.json',\n                    'Same actual source/PID-bound standalone DEV admission path')\n            admissions=[]\n            for _ in range(3):\n                context=self._family_guard();observed=sample_memory(self.pid,'DEV_actual_preimport_admission')\n                require(observed['paired_sampling_within_one_second'] and observed['worker_private_memory']['available'],\n                        'Actual DEV private before prospective remaining whole-worker budget')\n                remaining=max(full-observed['worker_private_memory']['private_bytes'],0)\n                admitted=fresh_admit(max(remaining+context['remaining_parent_and_allshim_allowance_bytes'],1),pid=self.pid)\n                require(admitted['passed'],'Three fresh whole DEV-family admissions before ML import')\n                admissions.append(admitted);time.sleep(.2)\n            write_json(folder/'owned_worker_admission.json',{'producer':self.producer,'request':self.config['request'],\n                'ROOT_execution_review':self.config['ROOT_execution_review'],'ROOT_resource_review':self.config['ROOT_resource_review'],\n                'whole_worker_fresh_three_sample_admission_passed':True,'whole_worker_forecast_bytes':full,\n                'configuration':self.configuration_binding,'parent_source':record(__file__),'samples':admissions})\n            self.command_state='autonomous_serial350DEV'\n        print(json.dumps({'actual_GPU_native_PID': self.pid, 'launcher_PID': self.child.pid, 'folder': str(folder)}), flush=True)")
replace("        self.command('finish_coherent_native_process')\n        deadline = time.monotonic()+60",
"        autonomous=self.config['worker_mode']=='serial350DEV_optimizer_free'\n        if not autonomous:self.command('finish_coherent_native_process')\n        # Standalone DEV has no worker_ready/command/finish protocol. Its real\n        # process0 names the inner report; the release envelope is separately required.\n        deadline=time.monotonic()+(self.config['maximum_GPU_child_wall_seconds'] if autonomous else 60)")
replace("        code = self.child.wait()\n        from asr18_owned_family_observation_v4 import wait_captured_dead",
"        code = self.child.wait()\n        if autonomous and code==0:\n            envelope=read_json(self.folder/'worker_completion.json')\n            require(envelope is not None and envelope['passed'] is True and envelope['producer']==self.producer\n                and envelope['worker_source']==self.config['worker_driver_source'] and envelope['source']==self.config['worker_driver_source']\n                and envelope['parent_source']==record(__file__) and envelope['worker_report']==record(self.completion_path)\n                and envelope['request']==self.config['request'] and len(envelope['phase_memory_records'])==4,\n                'Autonomous DEV actual release envelope plus SAME canonical inner report before process0')\n        from asr18_owned_family_observation_v4 import wait_captured_dead")
replace("def controller(config_path,review_path,folder):\n    from asr18_full_run_parent_controller_v1 import run",
"_ACTIVE_OWNER=None\n\ndef install_owner(owner):\n    global _ACTIVE_OWNER\n    require(_ACTIVE_OWNER is None,'One actual owner per process');_ACTIVE_OWNER=owner\n\ndef execute_owned_DEV(request_pin):\n    require(_ACTIVE_OWNER is not None,'Actual owner installed before DEV callback')\n    return _ACTIVE_OWNER.execute_DEV(request_pin)\n\ndef native_boundary_DEV(pointer):\n    require(_ACTIVE_OWNER is not None,'Actual owner installed before native boundary callback')\n    return _ACTIVE_OWNER.make_DEV_boundary(pointer)\n\ndef controller(config_path,review_path,folder):\n    from asr18_full_run_parent_controller_v2 import run")
replace("cleanup_script=Path(__file__).resolve(),storage=storage,fresh_admit=fresh_admit)",
"cleanup_script=Path(__file__).resolve(),storage=storage,fresh_admit=fresh_admit,\n        execute_owned_DEV=execute_owned_DEV,native_boundary_DEV=native_boundary_DEV,install_owner=install_owner)")
ast.parse(text);compile(text,str(new),'exec')
new.write_text(text,encoding='utf-8',newline='\n')
data=new.read_bytes();print({'path':str(new),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'scope':'AST_only_no_target_import'})
