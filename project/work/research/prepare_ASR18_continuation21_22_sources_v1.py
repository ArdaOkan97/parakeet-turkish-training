"""Source-only narrow continuation; preserve old source files and scientific state."""
import ast
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'


def replace(text,old,new):
    assert old in text,old
    return text.replace(old,new)


def write(name,text):
    p=S/name;assert not p.exists(),p;ast.parse(text);p.write_text(text,encoding='utf-8',newline='\n')


context=(S/'bpe_final_DATA_schedule_context_v1.py').read_text(encoding='utf-8')
context=replace(context,"require(1 <= pointer['committed_updates'] <= 20 and type(maximum_updates) is int\n            and 1 <= maximum_updates <= 20 and pointer['committed_updates']+maximum_updates <= 40,",
    "require(20 <= pointer['committed_updates'] <= 21 and type(maximum_updates) is int\n            and maximum_updates == 1 and pointer['committed_updates']+maximum_updates <= 22,")
context=replace(context,"'Only the second bounded window, actual source count and<=40 total'",
    "'Only actual20→21→22 one-update diagnostic continuation, no scientific reset'")
context=replace(context,"'ASR18_actual_checkpoint_second_window_preparation_v1'",
    "'ASR18_actual_checkpoint21_22_diagnostic_window_preparation_v1'")
write('bpe_final_DATA_schedule_context_v2.py',context)

engine=(S/'asr18_full_DATA_native_window_v5.py').read_text(encoding='utf-8')
engine=replace(engine,'bpe_final_DATA_schedule_context_v1','bpe_final_DATA_schedule_context_v2')
engine=replace(engine,"        self.schedule = self.context.new_schedule(); self.forecast = CalibratedGeometryForecast(**config['forecast_calibration'], guard=guard)",
    "        from asr18_verified_pipeline_ancestor_v1 import verify_ancestor\n        verify_ancestor(config['continuation_ancestor'],guard=guard)\n        from fastmss_phase_aware_feature_window_v4 import read_bound\n        session=read_bound(config['session'],guard)\n        require(session['initial_optimizer_updates']==20 and session['ancestor']==config['continuation_ancestor']\n                and config['maximum_total_native_updates']==22 and config['maximum_updates_per_window']==1,\n                'Explicit two one-update windows21/22 from actual20; no reset or fullwindow claim')\n        self.schedule = self.context.new_schedule(); self.forecast = CalibratedGeometryForecast(**config['forecast_calibration'], guard=guard)")
engine=replace(engine,"                and 1 <= pointer['committed_updates'] <= 20, 'Actual first-window coherent source required')",
    "                and type(pointer['committed_updates']) is int and 20 <= pointer['committed_updates'] <= 21, 'Actual20 ancestor or actual21 measured fallback required')\n        from asr18_verified_pipeline_ancestor_v1 import verify_ancestor\n        ancestor=verify_ancestor(self.config['continuation_ancestor'],guard=self.guard)\n        caller=pointer['identity']['caller_identity']\n        if pointer['committed_updates']==20:\n            require(pointer_binding==self.config['continuation_ancestor']['checkpoint_pointer']\n                    and digest(pointer)==digest(ancestor)\n                    and caller['session']==self.config['continuation_ancestor']['session'],\n                    'Exact original20 scientific identity before any live tensor mutation')\n        else:\n            from fastmss_owned_process_closure_v2 import verify as verify_process\n            boundary=read_bound(self.config['restore_parent_boundary'],self.guard)\n            actual0=read_bound(boundary['actual_native_closed'],self.guard)\n            verify_process(actual0,kind='ASR18_owned_GPU_process_completion_v1',owner='/root/memory_diagnosis',guard=self.guard)\n            require(boundary['coherent_checkpoint_pointer']==pointer_binding\n                    and caller['session']==self.config['session'] and caller['engine']==record(__file__)\n                    and caller['ancestor_checkpoint']==self.config['continuation_ancestor']['checkpoint_pointer']\n                    and actual0['parent_source']==self.config['parent_driver_source'],\n                    'Actual closed21 same-session fallback before any live tensor mutation')")
engine=replace(engine,"require(caller['session'] == self.config['session'], 'Fallback cannot change source-bound diagnostic session')",
    "require(caller['session'] == (self.config['continuation_ancestor']['session'] if pointer['committed_updates']==20 else self.config['session']),\n                'Scientific ancestor retained; only explicitly reviewed diagnostic orchestration session changes')")
engine=replace(engine,"self.process_initial_updates = self.optimizer.updates; self.window_count = 1",
    "self.process_initial_updates = self.optimizer.updates; self.window_count = self.optimizer.updates-20")
engine=replace(engine,"'session':self.config['session'],'config':record(self.config['_path']),'engine':record(__file__),",
    "'session':self.config['session'],'config':record(self.config['_path']),'engine':record(__file__),\n            'ancestor_checkpoint':self.config['continuation_ancestor']['checkpoint_pointer'],\n            'ancestor_actual_TOOL0':self.config['continuation_ancestor']['actual_TOOL0'],")
write('asr18_full_DATA_native_window_v6.py',engine)

recovery=(S/'asr18_deferred_GPU_recovery_v1.py').read_text(encoding='utf-8')
recovery=replace(recovery,'bpe_final_DATA_schedule_context_v1','bpe_final_DATA_schedule_context_v2')
recovery=replace(recovery,"'ASR18_deferred_all_checkpoint_GPU_recovery_requests_v1'","'ASR18_continuation21_22_all_new_checkpoint_GPU_recovery_requests_v2'")
recovery=replace(recovery,"            and 1 <= value['expected_global_new_updates'] <= 40",
    "            and value['expected_global_new_updates']==22\n            and type(value['initial_global_updates']) is int and value['initial_global_updates']==20\n            and type(value['expected_new_updates']) is int and value['expected_new_updates']==2")
recovery=replace(recovery,"    reported = []; actual_updates = 0",
    "    from asr18_verified_pipeline_ancestor_v1 import verify_ancestor\n    verify_ancestor(value['ancestor'],guard=guard)\n    declared_session=read_bound(session,guard)\n    require(declared_session['initial_optimizer_updates']==20 and declared_session['ancestor']==value['ancestor'],\n            'Exact inherited global20/session prefix; no scientific state reset')\n    reported = []; actual_updates = 0")
recovery=replace(recovery,"and actual_updates == value['expected_global_new_updates']",
    "and actual_updates == value['expected_new_updates']\n            and actual_updates+value['initial_global_updates']==value['expected_global_new_updates']")
recovery=replace(recovery,"require(total == actual_updates and 1 <= len(checkpoints) <= 6",
    "require(total == actual_updates and len(checkpoints)==2")
recovery=replace(recovery,"require(type(update) is int and 1 <= update <= actual_updates",
    "require(type(update) is int and value['initial_global_updates'] < update <= value['expected_global_new_updates']\n                and caller['ancestor_checkpoint']==value['ancestor']['checkpoint_pointer']\n                and caller['ancestor_actual_TOOL0']==value['ancestor']['actual_TOOL0']")
recovery=replace(recovery,"require(indices == sorted(set(indices)) and indices[-1] == actual_updates,",
    "require(indices == [21,22],")
recovery=replace(recovery,"'ASR18_actual_all_deferred_checkpoints_fresh_GPU_recovery_v1'",
    "'ASR18_actual_continuation21_22_all_new_captures_fresh_GPU_recovery_v2'")
recovery=replace(recovery,"'actual_global_native_updates':requests['expected_global_new_updates'],",
    "'actual_global_native_updates':requests['expected_global_new_updates'],\n        'initial_inherited_global_updates':20,'actual_new_training_updates_in_source_chain':2,\n        'ancestor':requests['ancestor'],'all_old_ancestor_captures_recovered_in_prior_actual0_only':True,")
write('asr18_deferred_GPU_recovery_v2.py',recovery)

runtime=(S/'asr18_restore_only_runtime_v1.py').read_text(encoding='utf-8')
runtime=replace(runtime,'asr18_deferred_GPU_recovery_v1','asr18_deferred_GPU_recovery_v2')
runtime=replace(runtime,'bpe_final_DATA_schedule_context_v1','bpe_final_DATA_schedule_context_v2')
write('asr18_restore_only_runtime_v2.py',runtime)
