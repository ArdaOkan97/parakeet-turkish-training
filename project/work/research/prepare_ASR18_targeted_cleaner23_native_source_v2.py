"""Derive one NEW inactive cleaner engine from the reviewed frozen engine5."""
import ast,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
source=S/'asr18_full_DATA_native_window_v5.py';target=S/'asr18_targeted_cleaner23_native_v1.py'
assert not target.exists() and hashlib.sha256(source.read_bytes()).hexdigest()=='971523f133034717fe621e799c597e559dc241e36a543347232140f1b53b4657'
text=source.read_text(encoding='utf-8-sig')
def replace(old,new):
    global text
    assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
replace('from bpe_training_schedule_v1 import WEAK','from bpe_training_schedule_v1 import CLEANER')
replace('from bpe_final_DATA_schedule_context_v1 import ScheduleContext, normalize_schedule_state',
        'from asr18_targeted_cleaner_schedule_context_v1 import CleanerContext, normalize_diagnostic_state')
replace('self.context = ScheduleContext(config[\'planning_arguments\'], guard=guard)',
        "self.context = CleanerContext(config['planning_arguments'],config['cleaner_ancestor'],guard=guard)")
replace('Only an explicitly bounded full-DATA diagnostic; no production launch',
        'Only an explicitly bounded targeted cleaner23 diagnostic; no production launch')
replace('self.schedule = self.context.new_schedule(); self.forecast',
        "require(config['maximum_windows']==1 and config['maximum_total_native_updates']==23 and config['maximum_updates_per_window']==1,'Exactly one NEW cleaner23 diagnostic')\n        self.schedule = self.context.new_schedule(); self.forecast")
begin=text.index('    def restore_parent(');end=text.index('    def _checkpoint(',begin)
restore='''    def restore_parent(self, pointer_binding, *, augmentation_identity):
        from fastmss_phase_aware_feature_window_v4 import read_bound
        from asr18_verified_pipeline_ancestor_v2 import verify_ancestor
        pointer=read_bound(pointer_binding,self.guard)
        ancestor=verify_ancestor(self.config['cleaner_ancestor'],guard=self.guard)
        require(pointer_binding==self.config['cleaner_ancestor']['checkpoint_pointer']
                and digest(pointer)==digest(ancestor) and pointer['committed_updates']==22
                and pointer['diagnostic_only'] is True,'Exact actual22 before any live tensor mutation')
        self.binding=MetadataCheckpointBinding(augmentation_identity,guard=self.guard)
        def operation():
            from asr18_streamed_persistence_fingerprint_v1 import normalize_fingerprint
            from asr18_checkpoint_next_draw_v1 import next_draw_packet as original_next_draw
            from bpe_final_DATA_schedule_context_v2 import normalize_schedule_state
            state,journal,branch=restore_state(pointer['checkpoint'],torch=self.torch,model=self.model,optimizer=self.optimizer,
                identity=pointer['identity'],augmentation=self.binding,
                new_journal_path=self.folder/'restored_actual22_authoritative_prefix.jsonl',diagnostic_only=True,guard=self.guard)
            require(self.optimizer.updates==22 and digest(state)==pointer['sampler_state_sha256']
                    and digest(state)==digest(self.context.original_state),'Exact actual22 original fullpopulation schedule retained')
            caller=pointer['identity']['caller_identity']
            expected=normalize_fingerprint(read_bound(caller['pre_save_fingerprint'],self.guard))
            require_exact_live_restore(expected,live_state_fingerprint(self.optimizer,torch=self.torch,guard=self.guard))
            original=self.context.source.new_schedule();original.load_state_dict(normalize_schedule_state(state))
            source_next=original_next_draw(original,self.context.source,optimizer_updates=22,guard=self.guard)
            require(digest(source_next)==digest(read_bound(caller['next_draw_packet'],self.guard)),
                    'Restored22 actual original next fullsource draw/replay must match PREsave before diagnostic fork')
            self.schedule=self.context.new_schedule()
            require(self.schedule.committed_updates()==self.schedule.replay.updates==22
                    and self.schedule.cleaner.draws==0
                    and digest(self.schedule.state_dict())==self.config['initial_cleaner_diagnostic_sampler_sha256'],
                    'Explicit derived diagnostic cleaner state, with unchanged actual22 ancestry and exact carried replay')
            require_exact_live_restore(expected,live_state_fingerprint(self.optimizer,torch=self.torch,guard=self.guard))
            return state,journal,branch
        (state,journal,branch),memory=self._run_phase('restore_actual22_before_targeted_cleaner',
            phase='startup_restore',increment=350_000_000,operation=operation)
        self.journal=journal;self.forecast.load_state_dict(pointer['identity']['caller_identity']['native_forecast_high_water'])
        self.process_initial_updates=22;self.window_count=0
        write_json(self.folder/'actual_parent_restore.json',{'passed':True,'pointer':pointer_binding,'branch':branch,
            'memory':memory,'actual_original_state_sha256':digest(state),'actual_diagnostic_initial_state_sha256':digest(self.schedule.state_dict()),
            'all207_threeAdam_all4RNG_match_actual_pre_save':True,'original_source_next_draw_packet_match':True,
            'original_broad_and_production_cleaner_cursors_unchanged':True,'production_phase_advance_claimed':False,
            'actual_completed_broad_passes':0,'production_enabled':False})

'''
text=text[:begin]+restore+text[end:]
replace('from asr18_checkpoint_next_draw_v1 import next_draw_packet\n',
        'from asr18_targeted_cleaner_schedule_context_v1 import completed_diagnostic_next_draw_packet as next_draw_packet\n')
replace("'feature_window':window_binding,'feature_owner':owner_binding,",
        "'feature_window':window_binding,'feature_owner':owner_binding,\n            'targeted_cleaner_diagnostic':True,'ancestor_checkpoint':self.config['cleaner_ancestor']['checkpoint_pointer'],\n            'ancestor_actual_TOOL0':self.config['cleaner_ancestor']['actual_TOOL0'],\n            'cleaner_schedule_adapter':record(Path(__file__).with_name('asr18_targeted_cleaner_schedule_context_v1.py')),\n            'production_phase_advance_claimed':False,")
replace("plan['phase'] == self.schedule.phase == WEAK","plan['phase'] == self.schedule.phase == CLEANER")
replace("'window_index': self.window_count-1, 'actual_new_updates'",
        "'diagnostic_only':True,'targeted_cleaner23':True,'production_phase_advance_claimed':False,\n            'window_index': self.window_count-1, 'actual_new_updates'")
ast.parse(text,filename=str(target))
with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
print(str(target),len(target.read_bytes()),hashlib.sha256(target.read_bytes()).hexdigest(),flush=True)

