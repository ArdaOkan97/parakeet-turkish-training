"""Preserve frozen v1; harden new native/monitor/checkpoint exception evidence."""
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=Path(__file__).resolve().parent
p=S/'asr18_cleaner_precomputed_probe_v1.py';q=p.with_name('asr18_cleaner_precomputed_probe_v2.py');t=p.read_text()
t=t.replace('from asr18_feature_window_native_probe_v1 import NativePeak','from asr18_feature_window_native_probe_v1 import NativePeak\nfrom asr18_cleaner_native_frontier_evidence_v1 import publish as publish_frontier')
needle="    finally:\n        memory=peak.finish();memory['wall_seconds']=time.monotonic()-began;write_json(folder/'native_phase_21.json',memory)"
replacement="""        publish_frontier(folder,optimizer,native=native,committed=False)
    except BaseException as exc:
        publish_frontier(folder,optimizer,error=repr(exc))
        raise
    finally:
        memory=peak.finish();memory['wall_seconds']=time.monotonic()-began;write_json(folder/'native_phase_21.json',memory)"""
assert needle in t;t=t.replace(needle,replacement)
needle="    write_json(folder/'native_progress.json',{'completed_updates':21,'native_updates_this_child':1,'clocks':[64271,2021,62271],\n        'journal_sampler_committed':False,'history':[native],'diagnostic_only':True})"
assert needle in t;t=t.replace(needle,'    publish_frontier(folder,optimizer,native=native,committed=False)')
needle="    write_json(folder/'native_progress.json',{'completed_updates':21,'native_updates_this_child':1,'clocks':[64271,2021,62271],\n        'journal_sampler_committed':True,'history':[native],'diagnostic_only':True})"
assert needle in t;t=t.replace(needle,'    publish_frontier(folder,optimizer,native=native,committed=True)')
start=t.index("    saved=save_state(folder/'after_update21.state.json'");end=t.index("    require(record(saved['path'])==saved",start)
body=t[start:end];wrapped="    checkpoint_before=sample_memory(os.getpid(),'cleaner_checkpoint21_before')['worker_private_memory'];require(checkpoint_before['available'],'Available checkpoint private baseline')\n    checkpoint_peak=NativePeak();checkpoint_peak.start()\n    try:\n"+''.join('    '+line+'\n' for line in body.splitlines())+"    finally:\n        checkpoint_memory=checkpoint_peak.finish()\n    checkpoint_covered=checkpoint_memory['private_peak_bytes']<=checkpoint_before['private_bytes']+64_000_000\n    write_json(folder/'checkpoint21_write_memory.json',{'before':checkpoint_before['private_bytes'],'phase':checkpoint_memory,'incremental_budget_bytes':64_000_000,'passed':checkpoint_covered})\n    require(checkpoint_covered,'Checkpoint21 private write forecast missed; preserve returned checkpoint as review-required')\n"
t=t[:start]+wrapped+t[end:]
start=t.index("    after,restored_journal,own_branch=restore_state(saved");end=t.index("    require(after==state",start);body=t[start:end]
wrapped="    restore_before=sample_memory(os.getpid(),'cleaner_checkpoint21_restore_before')['worker_private_memory'];require(restore_before['available'],'Available restore private baseline')\n    restore_peak=NativePeak();restore_peak.start()\n    try:\n"+''.join('    '+line+'\n' for line in body.splitlines())+"    finally:\n        restore_memory=restore_peak.finish()\n    restore_covered=restore_memory['private_peak_bytes']<=restore_before['private_bytes']+350_000_000\n    write_json(folder/'checkpoint21_restore_memory.json',{'before':restore_before['private_bytes'],'phase':restore_memory,'incremental_budget_bytes':350_000_000,'passed':restore_covered})\n    require(restore_covered,'Checkpoint21 restore private forecast missed; no prospective coverage approval')\n"
t=t[:start]+wrapped+t[end:]
t=t.replace("'native_memory':memory,'forecast_covered_actual_clean21':True", "'native_memory':memory,'checkpoint_memory':checkpoint_memory,'restore_memory':restore_memory,'all_native_write_restore_private_forecasts_covered':True,'forecast_covered_actual_clean21':True")
assert not q.exists();q.write_text(t,encoding='utf-8',newline='\n')
p=R/'prepare_ASR18_cleaner_hook_v1.py';q=p.with_name('prepare_ASR18_cleaner_hook_v2.py');t=p.read_text()
for name in ('run_ASR18_cleaner_hook','asr18_cleaner_precomputed_probe','asr18_cleaner_hook'):
    t=t.replace(name+'_v1',name+'_v2')
t=t.replace("text=text.replace('asr18_feature_window_control_v1.disabled.json'", "text=text.replace(\"progress['completed_updates']>0\",\"progress['native_updates_this_child']>0\")\n    text=text.replace('asr18_feature_window_control_v1.disabled.json'")
t=t.replace('production_cleaner_forecast_coordinator_review_manifest_v1.json','production_cleaner_forecast_coordinator_review_manifest_v2.json')
t=t.replace('ASR18_frozen_clean1_prospective_forecast_coordinator_source_request_v1','ASR18_frozen_clean1_prospective_forecast_coordinator_source_request_v2')
t=t.replace("A/'geometry_forecast_CPU_actual_02/report.json'", "A/'native_frontier_CPU_actual_01/report.json',A/'native_frontier_CPU_guard_01/closure.json',A/'geometry_forecast_CPU_actual_02/report.json'")
assert not q.exists();q.write_text(t,encoding='utf-8',newline='\n')
