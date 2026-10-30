"""Import/typed-scope successors only; no numerical runtime is executed."""
import ast,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
text=(S/'asr18_deferred_GPU_recovery_v2.py').read_text(encoding='utf-8-sig')
changes={
 'from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state':'from asr18_targeted_cleaner_schedule_context_v1 import normalize_diagnostic_state as normalize_schedule_state',
 'from asr18_checkpoint_next_draw_v1 import next_draw_packet':'from asr18_targeted_cleaner_schedule_context_v1 import completed_diagnostic_next_draw_packet as next_draw_packet',
 'ASR18_continuation21_22_all_new_checkpoint_GPU_recovery_requests_v2':'ASR18_targeted_cleaner23_all_new_checkpoint_GPU_recovery_requests_v1',
 "value['expected_global_new_updates']==22":"value['expected_global_new_updates']==23",
 "value['initial_global_updates']==20":"value['initial_global_updates']==22",
 "value['expected_new_updates']==2":"value['expected_new_updates']==1",
 "1 <= len(value['native_process_completions']) <= 2":"len(value['native_process_completions']) == 1",
 "1 <= len(value['native_window_completions']) <= 2":"len(value['native_window_completions']) == 1",
 'from asr18_verified_pipeline_ancestor_v1 import verify_ancestor':'from asr18_verified_pipeline_ancestor_v2 import verify_ancestor',
 "declared_session['initial_optimizer_updates']==20":"declared_session['initial_optimizer_updates']==22",
 'len(checkpoints)==2':'len(checkpoints)==1',
 'indices == [21,22]':'indices == [23]',
 'ASR18_actual_continuation21_22_all_new_captures_fresh_GPU_recovery_v2':'ASR18_actual_targeted_cleaner23_all_new_captures_fresh_GPU_recovery_v1',
 "'initial_inherited_global_updates':20":"'initial_inherited_global_updates':22",
 "'actual_new_training_updates_in_source_chain':2":"'actual_new_training_updates_in_source_chain':1",
 'Exact inherited global20/session prefix':'Exact inherited global22/session prefix',
 'Every actual1/10/final checkpoint is mandatory':'The actual new cleaner23 checkpoint is mandatory',
}
for old,new in changes.items():
    assert old in text,old;text=text.replace(old,new)
text=text.replace("window['passed'] is True and window['exact_GPU_restores_deferred'] is True", "window['passed'] is True and window['targeted_cleaner23'] is True and window['production_phase_advance_claimed'] is False\n                and window['exact_GPU_restores_deferred'] is True")
target=S/'asr18_targeted_cleaner23_recovery_v1.py';assert not target.exists();ast.parse(text)
with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
runtime=(S/'asr18_restore_only_runtime_v2.py').read_text(encoding='utf-8-sig')
runtime=runtime.replace('from asr18_deferred_GPU_recovery_v2 import verified_requests,recover_all',
    'from asr18_targeted_cleaner23_recovery_v1 import verified_requests,recover_all')
runtime=runtime.replace('from bpe_final_DATA_schedule_context_v2 import ScheduleContext',
    'from asr18_targeted_cleaner_schedule_context_v1 import CleanerContext')
runtime=runtime.replace("context=ScheduleContext(config['planning_arguments'],guard=guard)",
    "context=CleanerContext(config['planning_arguments'],config['cleaner_ancestor'],guard=guard)")
target2=S/'asr18_targeted_cleaner23_restore_runtime_v1.py';assert not target2.exists();ast.parse(runtime)
with target2.open('x',encoding='utf-8',newline='\n') as stream:stream.write(runtime)
for path in (target,target2):print(str(path),len(path.read_bytes()),hashlib.sha256(path.read_bytes()).hexdigest(),flush=True)
