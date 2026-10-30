"""Finite prospective policy/metadata binding controls; no GPU or native calls."""
import argparse,json,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import CommitJournal
from atomic_io_retry_v1 import write_json
from asr18_calibrated_geometry_forecast_v2 import CalibratedGeometryForecast
from asr18_metadata_checkpoint_binding_v1 import MetadataCheckpointBinding
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1');F=A.parent/'FastMSS_integration_v1'
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only policy controls');f.mkdir(exist_ok=False)
    args={'summary_binding':record(A/'feature_window_native20_01/actual_sameprocess20_summary_v1.json'),
        'archive_binding':record(F/'weak_fixed14_feature_window20_durable_metadata_after_native20_v1.json'),
        'retirement_binding':record(F/'weak_fixed14_feature_window20_actual_retirement_v1.json'),
        'native_source_binding':record(S/'asr18_precomputed_native_v2.py')}
    policy=CalibratedGeometryForecast(**args)
    window_path=Path(r'E:\gpu_training_scratchpad\temp\ASR18_CPU_feature_windows_v1\actual_cleaner_feature_window_01\manifest.json')
    window=json.loads(window_path.read_text());update=json.loads(Path(window['updates'][0]['update']['path']).read_text())
    proposed=policy.budget(rows=update['rows'],current_private_bytes=2047344640,update_in_process=1)
    checks=[]
    def reject(call,name):
        try:call()
        except ValueError:checks.append(name);return
        raise ValueError('Expected rejection: '+name)
    require(proposed['incremental_bytes']>=2500000000 and proposed['absolute_private_forecast_bytes']==5496813440,'Conservative first activation and running high-water basis')
    checks.append('Exact real cleaner4row geometry fits calibrated envelope; no GPU acceptance claim')
    for n in range(1,21):
        native=A/'feature_window_native20_01';phase=json.loads((native/f'native_phase_{n}.json').read_text());old=json.loads((native/f'update_{n}_budget.json').read_text())
        archive=json.loads(Path(args['archive_binding']['path']).read_text());rows=archive['metadata'][f'update_{n-1:02}.json']['rows']
        prediction=policy.budget(rows=rows,current_private_bytes=old['current_private_bytes'],update_in_process=n)
        require(phase['private_peak_bytes']<=old['current_private_bytes']+prediction['incremental_bytes'],'Retrospective private calibration coverage')
        policy.observe(phase)
    checks.append('Retrospective all20 native peaks covered; no prospective success claim')
    bad=json.loads(json.dumps(update['rows']));bad[0]['encoder_frames']=policy.envelopes['student_tr']['encoder_frames']+1
    reject(lambda:policy.budget(rows=bad,current_private_bytes=1,update_in_process=1),'Out-of-envelope full row stops before draw; no exclusion')
    reject(lambda:policy.observe({'error':'intentional','joined':True,'available_sample_count':3,'private_peak_bytes':1}),'Monitor errors rejected')
    reject(lambda:policy.observe({'error':None,'joined':True,'available_sample_count':0,'private_peak_bytes':1}),'Unavailable native monitor rejected')
    peak=policy.high_water;policy.observe({'error':None,'joined':True,'available_sample_count':3,'private_peak_bytes':1});require(policy.high_water==peak,'Highwater cannot decrease');checks.append('Running native high-water never decreases')
    # Exact old augmentation prefix can be verified without constructing an
    # augmenter. The original producer environment identity is preserved.
    cp=json.loads((A/'feature_window_native20_01/after_update20.state.json').read_text())
    def decode(node):
        k,v=node
        if k=='scalar':return v
        if k in ('list','tuple'):
            out=[decode(x) for x in v];return tuple(out) if k=='tuple' else out
        require(k=='dict','Scalar metadata only');return {decode(k):decode(v) for k,v in v}
    augmentation=decode(next(v for k,v in cp['tree'][1] if k==['scalar','augmentation']))
    binding=MetadataCheckpointBinding(augmentation['identity'],guard=lambda:None)
    require(binding.checkpoint_binding(committed_journal=augmentation['committed_journal'])==augmentation,'Exact frozen checkpoint contract without augmenter')
    require(binding.verify_restore(augmentation,verify_journal_prefix=CommitJournal(augmentation['committed_journal']['path']).verify)==augmentation['identity'],'Actual authoritative prefix verified')
    checks.append('Actual20 frozen policy/source identity and saved journal prefix verified without augmenter')
    require(not any(n in sys.modules for n in ('torch','numpy','scipy','transformers','pyarrow')),'No model/numerical import');checks.append('Metadata-only checkpoint/forecast paths import no numerical runtime')
    write_json(f/'report.json',{'passed':True,'checks':checks,'source':record(__file__),'forecast_source':record(S/'asr18_calibrated_geometry_forecast_v2.py'),
        'metadata_binding_source':record(S/'asr18_metadata_checkpoint_binding_v1.py'),'calibration':args,'prospective_clean_window':record(window_path),
        'prospective_budget_at_control_baseline':proposed,'actual_GPU_or_native_updates':0,'prospective_actual_success_claimed':False,'production_ready':False})
    print(json.dumps({'passed':True,'checks':len(checks),'geometry_envelopes':policy.envelopes,'absolute_private_forecast':proposed['absolute_private_forecast_bytes']}))
if __name__=='__main__':main()
