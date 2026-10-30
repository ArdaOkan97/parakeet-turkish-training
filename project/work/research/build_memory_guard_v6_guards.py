from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'work/research'
text=(R/'test_memory_guard_v5_cpu.py').read_text(encoding='utf-8')
text=text.replace('launch_with_memory_guard_v5','launch_with_memory_guard_v6').replace('memory_guard_v5_final','memory_guard_v6_final')
text=text.replace('import tempfile\n','import tempfile\nimport threading\n')
needle="with tempfile.TemporaryDirectory(dir=ROOT/'work/research', prefix='memory_v3_cpu_') as directory:"
insert="""# Actual short stdlib thread lifecycle and bounded mocked telemetry; no GPU.
arrived = threading.Event(); delivered = []
sampler = launch.LiveTelemetrySampler(123, lambda value: (delivered.append(value), arrived.set()), sample=sample)
sampler.start()
check('all-phase-thread-initial-observation', arrived.wait(1.) and len(delivered)==1)
sampler.stop_and_join()
check('all-phase-thread-stopped-and-joined', sampler.evidence()['stop_requested'] and sampler.evidence()['thread_join_completed'])
check('all-phase-thread-five-second-policy', sampler.evidence()['minimum_sampling_interval_seconds']==5.)
rejects('all-phase-thread-cannot-start-twice', sampler.start, RuntimeError)
for _ in range(130):sampler.sample_once()
check('all-phase-bounded128-records',len(sampler.samples)==128 and sampler.evidence()['samples_omitted']==3)
failed_sampler=launch.LiveTelemetrySampler(123, lambda _: None, sample=lambda *_: (_ for _ in ()).throw(OSError('sample fixture')))
failed_sampler.sample_once()
check('all-phase-sample-error-recorded',failed_sampler.failures==1)
failed_sink=launch.LiveTelemetrySampler(123,lambda _: (_ for _ in ()).throw(OSError('sink fixture')),sample=sample)
failed_sink.sample_once()
check('all-phase-publish-error-recorded',failed_sink.publish_failures==1 and failed_sink.failures==0)
not_started=launch.LiveTelemetrySampler(123,lambda _: None,sample=sample)
not_started.stop_and_join()
check('all-phase-not-started-cleanup-safe',not_started.evidence()['thread_join_completed'])
"""
text=text.replace(needle,insert+'\n'+needle)
text=text.replace("    check('source-class-restored',", "    check('all-phase-completed-runtime-joined',report['all_phase_owned_memory_telemetry']['thread_join_completed'] and report['all_phase_owned_memory_telemetry']['stop_requested'])\n    check('source-class-restored',")
text=text.replace("    check('class-restored-after-failure',", "    check('all-phase-failed-runtime-joined',failure['all_phase_owned_memory_telemetry']['thread_join_completed'])\n    check('class-restored-after-failure',")
p=R/'test_memory_guard_v6_cpu.py'
if p.exists():raise FileExistsError(p)
p.write_text(text,encoding='utf-8',newline='\n')
