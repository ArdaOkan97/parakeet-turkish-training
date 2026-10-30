"""Source-only narrow startup-domain/CUDA-unit successor, preserving all v1."""
import ast
from pathlib import Path

P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research'


def replace(text,old,new):
    assert old in text,old
    return text.replace(old,new)


def write(path,text):
    assert not path.exists(),path
    ast.parse(text)
    path.write_text(text,encoding='utf-8',newline='\n')


adapter=(S/'asr18_diagnostic_serial_DEV_v1.py').read_text(encoding='utf-8')
adapter=replace(adapter,'asr18_diagnostic_DEV_runtime_v1','asr18_diagnostic_DEV_runtime_v2')
write(S/'asr18_diagnostic_serial_DEV_v2.py',adapter)

runtime=(S/'asr18_diagnostic_DEV_runtime_v1.py').read_text(encoding='utf-8')
runtime=replace(runtime,'asr18_diagnostic_serial_DEV_v1','asr18_diagnostic_serial_DEV_v2')
runtime=replace(runtime,'asr18_diagnostic_DEV_runtime_v1','asr18_diagnostic_DEV_runtime_v2')
runtime=replace(runtime,"'CUDA_allocator_limit_GiB': 8.0, 'torch_threads': 4,",
    "'CUDA_allocator_limit_bytes': 8_000_000_000, 'capacity_argument_GB_decimal': 8.0, 'torch_threads': 4,")
runtime=replace(runtime,"capacity = validate_capacity(torch.cuda.get_device_properties(0).total_memory, 8.)",
    "capacity = validate_capacity(torch.cuda.get_device_properties(0).total_memory, 8.)\n    require(capacity['cap_bytes']==proposal['CUDA_allocator_limit_bytes']==8_000_000_000,'Exact decimal8GB allocator capacity')")
write(S/'asr18_diagnostic_DEV_runtime_v2.py',runtime)

parent=(R/'run_ASR18_diagnostic_serial_DEV_v1.py').read_text(encoding='utf-8')
for old,new in (('asr18_diagnostic_serial_DEV_v1','asr18_diagnostic_serial_DEV_v2'),
                ('asr18_diagnostic_DEV_runtime_v1','asr18_diagnostic_DEV_runtime_v2')):
    parent=replace(parent,old,new)
parent=replace(parent,"torch.cuda.max_memory_allocated()<=8*1024**3,'Unchanged8GiB actual CUDA allocated guard'",
    "torch.cuda.max_memory_allocated()<=8_000_000_000,'Unchanged decimal8GB actual CUDA allocated guard'")
parent=replace(parent,"CONFIG=read_json(config_path);check_bindings(CONFIG)\n    folder=Path(folder).resolve()",
    "CONFIG=read_json(config_path)\n    folder=Path(folder).resolve()")
parent=replace(parent,"WATCH=H.Watch(folder,5_450_000_000,worker=True)\n    code=1\n    write_json(folder/'entrypoint_boot.json',{'producer':identity(),'entrypoint_source':record(__file__)})",
    "write_json(folder/'entrypoint_boot.json',{'producer':identity(),'entrypoint_source':record(__file__)})\n    WATCH=H.Watch(folder,5_450_000_000,worker=True)\n    code=1")
parent=replace(parent,"try:\n        from asr18_diagnostic_DEV_runtime_v2 import run",
    "try:\n        check_bindings(CONFIG)\n        from asr18_diagnostic_DEV_runtime_v2 import run")
parent=replace(parent,"child=None;known={};native=None;launcher=None;code=1",
    "child=None;known={};native=None;launcher=None;runtime_verified=False;code=1")
old="""                if native is None and (output/'worker_boot_identity.json').exists():
                    boot=read_json(output/'worker_boot_identity.json');entry=read_json(output/'entrypoint_boot.json')
                    native=boot['producer']
                    require(same(entry['producer'],native) and same(entry['entrypoint_source'],record(__file__))
                            and same(boot['source'],c['source_bindings']['asr18_diagnostic_DEV_runtime_v2.py'])
                            and same(boot['request'],c['request']) and H.alive(native)
                            and (native['pid']==child.pid or native['ppid']==child.pid)
                            and (native['pid'],native['create_time']) in known,'Captured actual runtime/entrypoint PID/source')
"""
new="""                if native is None and (output/'entrypoint_boot.json').exists():
                    entry=read_json(output/'entrypoint_boot.json');native=entry['producer']
                    require(same(entry['entrypoint_source'],record(__file__)) and H.alive(native)
                            and (native['pid']==child.pid or native['ppid']==child.pid)
                            and (native['pid'],native['create_time']) in known,'Captured early native entrypoint identity/source')
                    # Native startup is covered by5.45GB from first recognition,
                    # never charged to the separate ALLshim64MB ceiling.
                if native is not None and not runtime_verified and (output/'worker_boot_identity.json').exists():
                    boot=read_json(output/'worker_boot_identity.json')
                    require(same(boot['producer'],native)
                            and same(boot['source'],c['source_bindings']['asr18_diagnostic_DEV_runtime_v2.py'])
                            and same(boot['request'],c['request']) and H.alive(native),
                            'Runtime boot MUST match already captured native before lease/admission')
                    runtime_verified=True
"""
parent=replace(parent,old,new)
parent=replace(parent,"require(actual==0 and native is not None and not any(H.alive(value) for value in known.values()),",
    "require(actual==0 and native is not None and runtime_verified and not any(H.alive(value) for value in known.values()),")
parent=replace(parent,"child=None;known={};native=None;launcher=None;runtime_verified=False;code=1",
    "child=None;known={};native=None;launcher=None;runtime_verified=False;output=None;code=1")
parent=replace(parent,"{'producer':identity(),'entrypoint_source':record(__file__)}",
    "{'producer':identity(),'entrypoint_source':record(__file__),'configuration':record(config_path),'request':CONFIG['request']}")
parent=replace(parent,"require(same(entry['entrypoint_source'],record(__file__)) and H.alive(native)",
    "require(same(entry['entrypoint_source'],record(__file__)) and same(entry['configuration'],record(config))\n                            and same(entry['request'],c['request']) and H.alive(native)")
parent=replace(parent,"launcher=identity(child.pid)","launcher=identity(child.pid)\n            known[(launcher['pid'],launcher['create_time'])]=launcher")
reconcile='''def reconcile_entrypoint(folder,child,known):
    """Even an already-closed launcher cannot hide its recorded live native."""
    if child is None:return True
    try:
        import psutil
        entry=read_json(Path(folder)/'entrypoint_boot.json');native=entry['producer']
        config_path=Path(folder).parent/'actual_worker_configuration.json'
        config=read_json(config_path)
        require(same(entry['entrypoint_source'],record(__file__))
                and same(entry['configuration'],record(config_path)) and same(entry['request'],config['request'])
                and (native['pid']==child.pid or native['ppid']==child.pid),'Exact independent early entrypoint provenance')
        known[(native['pid'],native['create_time'])]=native
        if H.alive(native):
            for process in [psutil.Process(native['pid']),*psutil.Process(native['pid']).children(recursive=True)]:
                try:value=identity(process.pid);known[(value['pid'],value['create_time'])]=value
                except psutil.NoSuchProcess:pass
        return True
    except BaseException as error:
        write_json(Path(folder).parent/'unresolved_early_owned_identity.json',
            {'error':repr(error),'source':record(__file__),'launcher_PID':child.pid,
             'captured_identities':list(known.values()),'idle_lease_release_permitted':False})
        return False


'''
parent=replace(parent,'def terminate_and_wait(known,child):',reconcile+'def terminate_and_wait(known,child):')
parent=replace(parent,"        terminate_and_wait(known,child)\n        # Original idle lease restored only after all captured exact identities died.\n        write_json(LEASE,prior)\n        watch.close(code)",
    "        resolved=reconcile_entrypoint(output,child,known)\n        terminate_and_wait(known,child)\n        if resolved:\n            write_json(LEASE,prior)\n        else:\n            unresolved=copy.deepcopy(prior)\n            unresolved.update(actual_GPU_running='unknown_owned_startup_identity_unresolved',\n                actual_owned_worker_pid=None,status='blocked_unresolved_owned_startup_identity',\n                unresolved_startup_evidence=str(folder/'unresolved_early_owned_identity.json'))\n            write_json(LEASE,unresolved)\n        watch.close(code)\n        require(resolved,'Early native identity unresolved; no idle lease release')")
write(R/'run_ASR18_diagnostic_serial_DEV_v2.py',parent)

builder=(R/'prepare_ASR18_diagnostic_serial_DEV_configuration_v1.py').read_text(encoding='utf-8')
for old,new in (('asr18_diagnostic_serial_DEV_v1','asr18_diagnostic_serial_DEV_v2'),
                ('asr18_diagnostic_DEV_runtime_v1','asr18_diagnostic_DEV_runtime_v2'),
                ('run_ASR18_diagnostic_serial_DEV_v1','run_ASR18_diagnostic_serial_DEV_v2'),
                ('diagnostic_optimizer_free_serial350DEV_configuration_01','diagnostic_optimizer_free_serial350DEV_configuration_02')):
    builder=replace(builder,old,new)
builder=replace(builder,"R/'prepare_ASR18_diagnostic_DEV_sources_v1.py',Path(__file__)",
    "R/'prepare_ASR18_diagnostic_DEV_sources_v1.py',R/'prepare_ASR18_diagnostic_DEV_successors_v2.py',Path(__file__)")
builder=replace(builder,"'source-bound actual parent+runtime entry identity,128MB+ALL64MBshims counted',",
    "'early native entrypoint identity before validation; matching runtime boot before lease/admission; native5.45GB separate from ALLshims64MB',\n            'exact decimal8GB allocator/peak gate; source validate_capacity(8.) uses8e9, not GiB',")
write(R/'prepare_ASR18_diagnostic_serial_DEV_configuration_v2.py',builder)
