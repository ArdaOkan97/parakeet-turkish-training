"""One tiny real-cleaner diagnostic prefix; never full-stage coverage."""
from bpe_broad_production_state_v1 import CompletePassCursor,require
class DiagnosticCleanerCursor:
    def __init__(self,*,ordinals,identity,seed):
        require(type(ordinals) is list and 1<=len(ordinals)<=14 and len(set(ordinals))==len(ordinals)
            and all(type(v) is int and 1<=v<=352768 for v in ordinals),'Bounded real cleaner ordinals required')
        self.ordinals=list(ordinals);self.inner=CompletePassCursor(population=len(ordinals),identity=identity,seed=seed,passes=1)
        self.population=len(ordinals)
    @property
    def draws(self):return self.inner.draws
    @property
    def remaining(self):return self.inner.remaining
    def preview(self,count):return [{**v,'ordinal':self.ordinals[v['ordinal']-1]} for v in self.inner.preview(count)]
    def commit(self,prepared):
        require(prepared==self.preview(len(prepared)),'Exact real cleaner diagnostic prefix required')
        self.inner.commit([{**v,'ordinal':self.ordinals.index(v['ordinal'])+1} for v in prepared])
    def state_dict(self):return {'kind':'ASR18_one_bounded_real_cleaner_diagnostic_cursor_v1','ordinals':list(self.ordinals),'inner':self.inner.state_dict(),'production_cleaner_passes':0}
    def load_state_dict(self,value):
        require(value['kind']=='ASR18_one_bounded_real_cleaner_diagnostic_cursor_v1' and value['ordinals']==self.ordinals and value['production_cleaner_passes']==0,'Exact diagnostic cleaner identity; no production phase claim')
        self.inner.load_state_dict(value['inner'])
