"""Metadata-only negative control: an old returned checkpoint relabeled current.

All optimizers/populations/checkpoint payloads are explicitly synthetic. A
successful fixture exit means the unsafe publication was REPRODUCED, never
that a real native checkpoint or production coordinator passed validation.
"""
from copy import deepcopy
import argparse,json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];R=Path(r'E:\gpu_training_scratchpad')
sys.path[:0]=[str(P/'outputs/parakeet_turkish'),str(P/'work/research')]
from bpe_training_schedule_v1 import TrainingSchedule
from bpe_production_window_coordinator_v2 import make_plan,WindowCoordinator
from bpe_broad_production_state_v1 import CommitJournal,digest
from bpe_broad_audio_v1 import record,require
from atomic_io_retry_v1 import write_json
from validate_ASR18_training_schedule_v1 import ReplayFixture,OptimizerFixture

def main(folder):
 out=Path(folder).resolve();require(out.is_relative_to(R),'E-only control');out.mkdir(exist_ok=False)
 require(os.environ['CUDA_VISIBLE_DEVICES']=='-1','Metadata-only control')
 def schedule():return TrainingSchedule(broad_population=15,broad_identity={'FIXTURE_ONLY':True},broad_mean_U=31.5,
  cleaner_population=352768,cleaner_identity={'FIXTURE_ONLY':True},cleaner_mean_U=11972886/352768,
  source_turkish_rng=('FIXTURE_ONLY',481),replay=ReplayFixture())
 def identity(phase,ordinal):return {'ordinal':ordinal,'source_id':phase+':'+str(ordinal),'language':'tr'}
 def rows(phase,draws,replay):return [identity(phase,d['ordinal']) for d in draws]+list(replay)
 population=out/'NOT_final_population.json';write_json(population,{'FIXTURE_ONLY':True})
 s=schedule();plan=make_plan(s,clone_schedule=schedule,resolve_identity=identity,resolve_rows=rows,
  feature_bytes_upper_bound=lambda phase,value:1000,finalized_population=record(population),verify_finalized_population=lambda bound:bound==record(population))
 prepared=out/'NOT_real_prepared_audio.json';write_json(prepared,{'FIXTURE_ONLY':True,'prepared_not_committed':True})
 opt=OptimizerFixture();journal=CommitJournal(out/'fixture_journal.jsonl');co=WindowCoordinator(s,opt,journal,plan,resolve_identity=identity)
 def native(planned,tx):
  opt.updates+=1;values=tx.TR_identity+list(tx.replay_rows)
  return {'phase':tx.phase,'update':opt.updates,'source_ids':[v['source_id'] for v in values],'languages':[v['language'] for v in values]}
 def consume():return co.consume(prepare_evidence=lambda planned:record(prepared),native_call=native,verify_preparation=lambda planned,bound:bound==record(prepared))
 consume();saved_state=deepcopy(s.state_dict());saved_prefix=journal.binding();saved_identity={'FIXTURE_ONLY':True,'identity_version':1}
 payload=out/'old_update1.tensors.bin';payload.write_bytes(b'FIXTURE_NO_MODEL_TENSORS')
 metadata=out/'old_update1.state.json'
 write_json(metadata,{'kind':'ASR18_streaming_named_tensor_checkpoint_v1','schema_version':1,'dynamic_module_serialization':False,
  'maximum_CPU_tensor_slice_bytes':4194304,'payload':record(payload),'FIXTURE_ONLY':True,
  'saved_identity':saved_identity,'saved_optimizer_updates':1,'saved_sampler_state_sha256':digest(saved_state),'saved_journal_prefix':saved_prefix})
 old=record(metadata);pointer=out/'latest_fixture.json'
 co.publish_checkpoint(write_coherent_checkpoint=lambda state,identity:old,pointer_path=pointer,checkpoint_identity=saved_identity)
 consume();current_state=deepcopy(s.state_dict());current_prefix=journal.binding();current_identity={'FIXTURE_ONLY':True,'identity_version':2}
 require(opt.updates==2 and digest(current_state)!=digest(saved_state) and current_prefix!=saved_prefix,'Two distinct hypothetical committed prefixes')
 actual=co.publish_checkpoint(write_coherent_checkpoint=lambda state,identity:old,pointer_path=pointer,checkpoint_identity=current_identity)
 published=json.loads(pointer.read_text());unchanged=opt.updates==2 and s.state_dict()==current_state and journal.binding()==current_prefix
 reproduced=(published['checkpoint']==old and published['committed_updates']==2 and published['identity']==current_identity and
  published['sampler_state_sha256']==digest(current_state) and published['journal_prefix']==current_prefix and unchanged)
 require(reproduced,'Expected current v2 to reproduce stale-return publication defect')
 require(not any(v in sys.modules for v in ('torch','numpy','scipy','transformers','soundfile','pyarrow')),'No numerical/model/audio import')
 write_json(out/'report.json',{'kind':'ASR18_metadata_only_stale_checkpoint_return_negative_control_v1','control_completed':True,
  'unsafe_publication_reproduced':True,'coordinator_publication_safety_passed':False,'source':record(__file__),
  'coordinator_source':record(P/'outputs/parakeet_turkish/bpe_production_window_coordinator_v2.py'),
  'old_fixture_checkpoint':old,'old_saved_fields':{'identity':saved_identity,'optimizer_updates':1,'sampler_state_sha256':digest(saved_state),'journal_prefix':saved_prefix},
  'published_pointer':published,'live_state_unchanged':unchanged,'current_optimizer_updates':2,
  'missing_checks_confirmed':['saved identity','saved optimizer update count','saved sampler-state digest','saved augmentation committed-journal prefix'],
  'same_structural_fixture_checkpoint_contract_as_existing26_controls':True,'real_named_tensor_checkpoint_or_native_restore_proved':False,
  'synthetic_optimizer_population_payload_only':True,'actual_native_updates':0,'actual_data_or_audio_jobs':0,'production_ready':False})
 print(json.dumps({'control_completed':True,'unsafe_publication_reproduced':True,'live_state_unchanged':unchanged,'actual_native_updates':0}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();main(a.folder)
