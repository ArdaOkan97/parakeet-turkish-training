"""Source-only adapt reviewed hard parent supervision to distinct full runtime."""
import ast
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=P/'work/research'
source=(R/'run_ASR18_hard_effective16_v3.py').read_text(encoding='utf-8-sig')
start=source.index('class ParentGuard:');end=source.index('class GPUService:')
guard='''class ParentGuard:
    """CPU-only owner: base128MB, read/index192MB, AUX-XZ384MB.

    Lifetime384MB remains a whole-process limit after an earlier archive phase;
    returning to base requires actual current<=128MB, never assuming release.
    """
    CAPS={'base':128_000_000,'read_index':192_000_000,'archive_aux':384_000_000}
    def __init__(self,folder):
        self.folder=folder;self.phase='base';self.current_cap=self.CAPS['base'];self.phase_depth=0
        self.watch=ResourceGuard();self.done=threading.Event();self.error=None;self.count=0;self.last=None;self.GPU=None
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start();deadline=time.monotonic()+3
        while self.count<3:
            require(self.thread.is_alive() and self.error is None and time.monotonic()<deadline,'Live parent guard before work');time.sleep(.05)
    def _sample(self):
        sample=sample_memory(os.getpid(),'full_owner_parent_'+self.phase);w=sample['worker_private_memory']
        require(sample['paired_sampling_within_one_second'] and w['available'] and sample['system_memory']['available'],'Fresh parent/system telemetry')
        sample['owned_current_and_lifetime']=private_lifetime_limit(384_000_000)
        require(w['private_bytes']<=self.current_cap,'Current parent exceeds exact metadata phase cap')
        reason,_=self.watch.observe(sample['system_memory'],w);require(reason is None,'Parent resource stop: '+str(reason))
        require(shutil.disk_usage(ROOT).free>100_000_000_000,'E100GB reserve')
        return sample
    def _run(self):
        try:
            while not self.done.is_set():
                sample=self._sample();self.count+=1;self.last=time.monotonic()
                write_json(self.folder/'parent_metadata_phase_live.json',{'source':record(__file__),'phase':self.phase,
                    'current_cap_bytes':self.current_cap,'whole_lifetime_cap_bytes':384_000_000,'sample':sample})
                self.done.wait(.2)
        except BaseException:self.error=traceback.format_exc()
    def check(self):
        require(self.error is None and self.thread.is_alive() and self.last is not None and time.monotonic()-self.last<2,'Parent guard unavailable/failed/stale')
        return self._sample()
    @__import__('contextlib').contextmanager
    def metadata_phase(self,phase):
        require(phase in ('read_index','archive_aux'),'Explicit reviewed metadata phase')
        if self.phase_depth:
            require(self.phase==phase or self.phase=='archive_aux','Cannot expand a nested phase silently')
            self.phase_depth+=1
            try:yield self.check()
            finally:self.phase_depth-=1
            return
        before=self.check();cap=self.CAPS[phase]
        # Global current already contains this parent and any resident native.
        # Reserve only unused parent phase cap plus actual remaining shim cap.
        extra=max(cap-before['worker_private_memory']['private_bytes'],0)+64_000_000
        if self.GPU is not None:
            self.GPU.check();require(self.GPU.command_state=='coherent_idle_CPU_preparation','No native operation during metadata compression/indexing')
            context=self.GPU._family_guard()
            extra=max(cap-before['worker_private_memory']['private_bytes'],0)+max(64_000_000-context['all_captured_GPU_shim_private_bytes'],0)
        admission=fresh_admit(max(extra,1));require(admission['passed'],'Prospective combined metadata phase denied')
        self.phase=phase;self.current_cap=cap;self.phase_depth=1
        write_json(self.folder/'parent_metadata_phase_admission.json',{'phase':phase,'source':record(__file__),'admission':admission,'before':before})
        try:yield self.check()
        finally:
            self.phase_depth=0
            value=self.check();require(value['worker_private_memory']['private_bytes']<=128_000_000,'Return-to-base actual current must fit128MB; no release assumption')
            self.phase='base';self.current_cap=self.CAPS['base'];self.check()
    def close(self):
        self.done.set();self.thread.join(timeout=2)
        require(not self.thread.is_alive() and self.error is None and self.count>=3,'Clean joined parent guard')


'''
source=source[:start]+guard+source[end:]
# Replace only complete concrete-controller tail; unchanged CPU/cleanup helpers remain.
source=source[:source.index('def controller(')]+'''def controller(config_path,review_path,folder):
    from asr18_full_run_parent_controller_v1 import run
    return run(config_path,review_path,folder,parent_source=record(__file__),ParentGuard=ParentGuard,
        GPUService=GPUService,run_CPU=run_CPU,cleanup_script=Path(__file__).resolve(),storage=storage,fresh_admit=fresh_admit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--review');p.add_argument('--folder',required=True)
    p.add_argument('--cleanup',action='store_true');p.add_argument('--request')
    a=p.parse_args();folder=scoped(a.folder)
    if a.cleanup:cleanup_child(a.request,folder)
    else:controller(a.config,a.review,folder)
'''
source=source.replace("self.configuration_binding = record(config_path)","self.configuration_binding = record(config_path)\n        self.completion_path=folder/('actual_current_recovery/current_recovery_report.json' if self.config['worker_mode']=='restore_only' else 'worker_completion.json')")
source=source.replace('from asr18_hard_effective16_recovery_v1 import verified_requests','from asr18_full_run_current_recovery_v2 import checked_request')
source=source.replace("verified_requests(self.config['deferred_recovery_requests'], session=self.config['session'],\n                parent_source=record(__file__), guard=parent_guard.check)","checked_request(self.config['current_recovery_request'],guard=parent_guard.check)")
source=source.replace("full+max(128_000_000-own,0)+64_000_000","full+max(384_000_000-own,0)+64_000_000")
old="self.child = subprocess.Popen([self.config['GPU_python']['path'], '-X', 'utf8', '-B', str(Path(__file__).resolve()),\n            '--worker' if self.config['worker_mode'] == 'resident_training' else '--recovery-worker',\n            '--config', str(config_path), '--folder', str(folder)],"
new="require(record(self.config['worker_driver_source']['path'])==self.config['worker_driver_source'],'Exact separate GPU runtime source')\n        self.child = subprocess.Popen([self.config['GPU_python']['path'], '-X', 'utf8', '-B', self.config['worker_driver_source']['path'],\n            '--config', str(config_path), '--folder', str(folder)],"
assert old in source;source=source.replace(old,new)
source=source.replace("while not (folder/'worker_boot_identity.json').exists()","while not (folder/'entrypoint_boot.json').exists()")
source=source.replace("boot = read_json(folder/'worker_boot_identity.json'); producer = boot['producer']","boot = read_json(folder/'entrypoint_boot.json'); producer = boot['producer']")
source=source.replace("boot['source'] == record(__file__) and producer == identity(producer['pid'])","boot['source'] == self.config['worker_driver_source'] and boot['configuration']==self.configuration_binding and boot['parent_source']==record(__file__) and producer == identity(producer['pid'])")
source=source.replace("self.folder/'worker_completion.json'","self.completion_path")
source=source.replace("worker_source=record(__file__),parent_source=record(__file__)","worker_source=self.config['worker_driver_source'],parent_source=record(__file__)")
source=source.replace("worker_source=record(__file__),\n                        parent_source=record(__file__)","worker_source=self.config['worker_driver_source'],\n                        parent_source=record(__file__)")
source=source.replace("report['worker_source']==record(__file__)","report['worker_source']==self.config['worker_driver_source']")
source=source.replace("worker_source=record(__file__), parent_source=record(__file__)","worker_source=self.config['worker_driver_source'], parent_source=record(__file__)")
source=source.replace("own_private<=128_000_000,'Separate parent128MB'","own_private<=self.parent_guard.current_cap,'Exact current parent metadata phase cap'")
source=source.replace("max(128_000_000-own_private,0)+max(64_000_000-shim,0)","max(self.parent_guard.current_cap-own_private,0)+max(64_000_000-shim,0)")
# Reconcile actual early entrypoint even when launcher exits before first loop.
needle="known = getattr(candidate, 'known', {})\n                producer = getattr(candidate, 'producer', None)"
replace="""known = getattr(candidate, 'known', {})
                early=read_json(candidate.folder/'entrypoint_boot.json')
                if early is not None:
                    require(early['source']==candidate.config['worker_driver_source'] and early['configuration']==candidate.configuration_binding
                        and early['parent_source']==record(__file__),'Exact early startup provenance before failure retirement')
                    entry=early['producer'];require(entry['pid']==child.pid or entry['ppid']==child.pid,'Owned early startup ancestry')
                    known[(entry['pid'],entry['create_time'])]=entry
                producer = getattr(candidate, 'producer', None)"""
assert needle in source;source=source.replace(needle,replace)
# Actual source runtime boot must also agree before any operation command.
needle="print(json.dumps({'actual_GPU_native_PID': self.pid, 'launcher_PID': self.child.pid, 'folder': str(folder)}), flush=True)"
replace="""deadline=time.monotonic()+30
        while not (folder/'worker_boot_identity.json').exists():
            self.check();require(time.monotonic()<deadline,'Bound runtime identity after source validation');time.sleep(.1)
        runtime_boot=read_json(folder/'worker_boot_identity.json')
        require(runtime_boot==boot,'Early entrypoint and runtime identity/configuration/source must agree')
        print(json.dumps({'actual_GPU_native_PID': self.pid, 'launcher_PID': self.child.pid, 'folder': str(folder)}), flush=True)"""
assert needle in source;source=source.replace(needle,replace)
# Failure retirement includes ALL exact captured children, not only the producer.
start=source.index('    def abort(self):');end=source.index('\n\ndef run_CPU',start)
source=source[:start]+'''    def abort(self):
        write_json(self.folder/'stop.json',{'reason':'owned_parent_failure_preserve_partial'})
        from asr18_owned_family_observation_v4 import terminate_known_family
        self.done.set();self.monitor.join(timeout=2)
        retired=terminate_known_family(self.child,self.launcher,self.known)
        write_json(self.folder/'failed_all_captured_identity_retirement.json',retired)
        self.log.close();self._release_lease()
'''+source[end:]
out=R/'run_ASR18_full_run_owner_v1.py';assert not out.exists();ast.parse(source);out.write_text(source,encoding='utf-8',newline='\r\n')
print(out)
