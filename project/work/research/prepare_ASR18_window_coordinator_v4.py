"""Exact canonical identity successor and isolated metadata drift controls."""
from pathlib import Path
import sys,json
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import require,record
def change(s,a,b):require(s.count(a)==1,'Exact source delta');return s.replace(a,b)
def main():
    old=S/'bpe_streaming_checkpoint_semantics_v1.py';target=S/'bpe_streaming_checkpoint_semantics_v2.py';require(not target.exists(),'Fresh semantic verifier')
    text=change(old.read_text(),'require(saved_identity==identity,','require(saved_identity==identity and digest(saved_identity)==digest(identity),');target.write_text(text)
    old=S/'bpe_production_window_coordinator_v3.py';target=S/'bpe_production_window_coordinator_v4.py';require(not target.exists(),'Fresh coordinator')
    text=old.read_text().replace('bpe_streaming_checkpoint_semantics_v1','bpe_streaming_checkpoint_semantics_v2')
    text=change(text,"require(updates==self.schedule.committed_updates() and digest(state)==last['after_sampler_sha256'],",
        "require(type(updates) is int and updates>0 and updates==self.schedule.committed_updates() and digest(state)==last['after_sampler_sha256'],")
    text=change(text,"require(self.optimizer.updates==updates and self.schedule.state_dict()==state and self.journal.binding()==journal\n            and self.index==index and digest(self.plan)==plan_digest and checkpoint_identity==caller_identity,",
        "require(type(self.optimizer.updates) is int and self.optimizer.updates==updates\n            and self.schedule.state_dict()==state and digest(self.schedule.state_dict())==digest(state)\n            and self.journal.binding()==journal and digest(self.journal.binding())==digest(journal)\n            and type(self.index) is int and self.index==index and digest(self.plan)==plan_digest\n            and checkpoint_identity==caller_identity and digest(checkpoint_identity)==digest(caller_identity),")
    target.write_text(text)
    old=R/'validate_ASR18_production_window_v4.py';target=R/'validate_ASR18_production_window_v5.py';require(not target.exists(),'Fresh identity drift controls')
    text=old.read_text().replace('bpe_production_window_coordinator_v3','bpe_production_window_coordinator_v4').replace('bpe_streaming_checkpoint_semantics_v1','bpe_streaming_checkpoint_semantics_v2')
    text=change(text,"    def writer(state,identity_value):\n        payload=f/('state_'+str(opt.updates)+'.bin');payload.write_bytes(b'FIXTURE_NOT_NATIVE_CHECKPOINT')\n        metadata=f/('state_'+str(opt.updates)+'.json')",
        "    writer_sequence=[0]\n    def writer(state,identity_value):\n        writer_sequence[0]+=1;stem='state_'+str(opt.updates)+'_attempt_'+str(writer_sequence[0])\n        payload=f/(stem+'.bin');payload.write_bytes(b'FIXTURE_NOT_NATIVE_CHECKPOINT')\n        metadata=f/(stem+'.json')")
    before="    for _ in range(2):co.consume(prepare_evidence=evidence,native_call=native,verify_preparation=verify)"
    after="""    from bpe_streaming_checkpoint_semantics_v2 import select
    def wrong_saved_identity(state,identity_value):
        good=writer(state,identity_value);document=json.loads(Path(good['path']).read_text())
        select(document['tree'],'identity','diagnostic_only')[:]=['scalar',1]
        wrong=f/'wrong_saved_identity_bool_int.json';write_json(wrong,document);return record(wrong)
    reject(lambda:co.publish_checkpoint(write_coherent_checkpoint=wrong_saved_identity,pointer_path=pointer,
        checkpoint_identity={'FIXTURE_ONLY':True},diagnostic_only=True),'Nested saved identity True-versus1 drift rejected')
    check(pointer.read_bytes()==before_stale and s.state_dict()==live_state and journal.binding()==live_prefix,
          'Saved identity drift preserves previous latest and live frontiers')
    external_identity={'caller_flag':True}
    def mutate_external_identity(state,identity_value):
        good=writer(state,identity_value);external_identity['caller_flag']=1;return good
    reject(lambda:co.publish_checkpoint(write_coherent_checkpoint=mutate_external_identity,pointer_path=pointer,
        checkpoint_identity=external_identity,diagnostic_only=True),'Caller identity mutation True-versus1 detected after writer')
    check(pointer.read_bytes()==before_stale and s.state_dict()==live_state,'Caller identity numeric drift cannot publish latest')
    for _ in range(2):co.consume(prepare_evidence=evidence,native_call=native,verify_preparation=verify)"""
    text=change(text,before,after)
    text=change(text,"        ('Stale saved Adam updates',('optimizer','updates'),['scalar',20]),",
        "        ('Nested actual identity True-versus1',('identity','declaration','diagnostic_only'),['scalar',1]),\n        ('Stale saved Adam updates',('optimizer','updates'),['scalar',20]),")
    target.write_text(text);print(json.dumps({'coordinator':record(S/'bpe_production_window_coordinator_v4.py'),'controls':record(target),'production_ready':False}))
if __name__=='__main__':main()
