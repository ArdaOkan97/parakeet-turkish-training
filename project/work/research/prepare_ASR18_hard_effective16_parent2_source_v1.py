"""Explicit parent128M/all-GPU-shim64M successor, no numerical source change."""
import ast
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'work/research'
text=(R/'run_ASR18_hard_effective16_v1.py').read_text(encoding='utf-8-sig')
def change(old,new):
    global text
    assert old in text,old;text=text.replace(old,new)
change('import argparse, copy, gc, json, os, secrets, shutil, subprocess, sys, threading, time, traceback',
       'import argparse, copy, ctypes, gc, json, os, secrets, shutil, subprocess, sys, threading, time, traceback\nfrom ctypes import wintypes')
change('from resource_watchdog_v2 import ResourceGuard, sample_memory',
       'from resource_watchdog_v2 import ResourceGuard, sample_memory, PROCESS_MEMORY_COUNTERS_EX')
insert='''def private_lifetime_limit(cap):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    value=PROCESS_MEMORY_COUNTERS_EX();value.cb=ctypes.sizeof(value)
    require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb),'Owned current/lifetime telemetry')
    require(value.PrivateUsage<=cap and value.PeakPagefileUsage<=cap,'Owned current AND lifetime private limit')
    return {'current_private_bytes':int(value.PrivateUsage),'OS_lifetime_private_peak_bytes':int(value.PeakPagefileUsage)}

def checked_stage(folder,name,torch,*,cap,**extra):
    value=stage(folder,name,torch,**extra)
    require(value['worker_private_memory']['private_bytes']<=cap and value['worker_private_peak_bytes_lifetime']<=cap,
            'Actual current/lifetime private must fit separately admitted whole worker')
    require(value.get('cuda',{}).get('peak_allocated_bytes',0)<=8_000_000_000,'Unchanged decimal8GB CUDA ceiling')
    return value


'''
change('class ParentGuard:',insert+'class ParentGuard:')
change("reason, detail = self.watch.observe(sample['system_memory'], worker)",
       "sample['owned_current_and_lifetime']=private_lifetime_limit(128_000_000)\n                reason, detail = self.watch.observe(sample['system_memory'], worker)")
change("worker['private_bytes'] < 2_000_000_000", "worker['private_bytes'] <= 128_000_000")
change("return sample_memory(os.getpid(), 'full_DATA_parent_callback')",
       "value=sample_memory(os.getpid(), 'full_DATA_parent_callback')\n        value['owned_current_and_lifetime']=private_lifetime_limit(128_000_000)\n        return value")
change('emit = lambda name, **extra: stage(folder, name, torch, **extra)',
       "emit = lambda name, **extra: checked_stage(folder,name,torch,cap=config['entire_worker_forecast']['entire_worker_private_forecast_bytes'],**extra)")
change("admission = fresh_admit(full); write_json(folder.parent/(folder.name+'_prelaunch_full_forecast.json'), admission)",
       "own=parent_guard.check()['worker_private_memory']['private_bytes']\n        admission = fresh_admit(full+max(128_000_000-own,0)+64_000_000); write_json(folder.parent/(folder.name+'_prelaunch_full_forecast.json'), admission)")
change("owned_family_sample(self.child, self.launcher, known=self.known, guard=parent_guard.check)",
       "startup_family=owned_family_sample(self.child, self.launcher, known=self.known, guard=parent_guard.check)\n            require(startup_family['private_bytes_sum']<=full+64_000_000 and all(item['private']['private_bytes']<=full for item in startup_family['measured_processes']),\n                    'Native startup domain unresolved: whole family bounded without mislabeling native as shim')")
change('    def _watch(self):\n', '''    def _family_guard(self):
        from asr18_owned_family_observation_v4 import owned_family_sample,same_process,reaped_native_success_pending
        own=self.parent_guard.check()
        try:
            family=owned_family_sample(self.child,self.launcher,known=self.known,guard=self.parent_guard.check)
        except ProcessLookupError:
            require(not same_process(self.producer),'Live native orphan cannot use a launcher-disappearance success transition')
            reaped_native_success_pending(self.producer,completion_path=self.folder/'worker_completion.json',
                worker_source=record(__file__),parent_source=record(__file__),launcher=self.child)
            family={'measured_processes':[],'launcher_disappearance_is_not_actual_wait0':True}
        measured={(part['identity']['pid'],part['identity']['create_time']):part for part in family['measured_processes']}
        # Count ALL captured still-live identities, including a reparented/draining shim.
        for key,identity in list(self.known.items()):
            if key in measured or not same_process(identity):continue
            value=sample_memory(identity['pid'],'hard_current_captured_family')['worker_private_memory']
            if not value['available'] or not same_process(identity):
                require(not same_process(identity),'Live captured family memory unavailable');continue
            measured[key]={'identity':identity,'private':value}
        native=(self.producer['pid'],self.producer['create_time'])
        shim=0
        for key,part in measured.items():
            require(part['private']['available'] and type(part['private']['private_bytes']) is int
                    and part['private']['private_bytes']>0,'Actual available exact captured family bytes')
            if key==native:
                require(part['private']['private_bytes']<=self.config['entire_worker_forecast']['entire_worker_private_forecast_bytes'],
                        'Actual native current private exceeds its whole worker')
            else:shim+=part['private']['private_bytes']
        require(shim<=64_000_000,'Continuously summed ALL exact GPU shim private cap64MB')
        own_private=own['worker_private_memory']['private_bytes'];require(own_private<=128_000_000,'Separate parent128MB')
        result={'parent':own,'all_captured_GPU_shim_private_bytes':shim,
            'remaining_parent_and_allshim_allowance_bytes':max(128_000_000-own_private,0)+max(64_000_000-shim,0),
            'native_identity':self.producer,'measured_processes':list(measured.values()),
            'source':record(__file__),'configuration':record(self.config['_path']) if '_path' in self.config else None,
            'observed_monotonic':time.monotonic(),'native_excluded_from_shim_sum':True}
        write_json(self.folder/'exact_GPU_family_current.json',result)
        return result

    def _watch(self):
''')
change("self.parent_guard.check(); sample = sample_memory(self.pid, 'full_DATA_GPU_parent_fresh')",
       "family_context=self._family_guard(); sample = sample_memory(self.pid, 'full_DATA_GPU_parent_fresh')")
change("result = fresh_admit(value['incremental_private_budget_bytes'], pid=self.pid)",
       "family_context=self._family_guard()\n                        require(time.monotonic()-family_context['observed_monotonic']<=1,'Fresh exact owned-family context')\n                        result = fresh_admit(value['incremental_private_budget_bytes']+family_context['remaining_parent_and_allshim_allowance_bytes'], pid=self.pid)")
change("'fresh_sample': result['sample'], 'incremental_budget_bytes': value['incremental_private_budget_bytes'],",
       "'fresh_sample': result['sample'], 'exact_family_context':family_context, 'incremental_budget_bytes': value['incremental_private_budget_bytes'],")
change('stage=stage,admit_stage=admit_stage)',
       'stage=lambda folder,name,torch,**extra:checked_stage(folder,name,torch,cap=3_350_000_000,**extra),admit_stage=admit_stage)')
path=R/'run_ASR18_hard_effective16_v2.py';ast.parse(text,filename=str(path));assert not path.exists()
with path.open('x',encoding='utf8',newline='\n') as stream:stream.write(text)
print(str(path),flush=True)
