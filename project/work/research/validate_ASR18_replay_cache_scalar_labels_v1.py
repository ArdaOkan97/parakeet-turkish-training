"""Actual replay cache labels against independent CPU Torch+PCM references."""
import argparse,hashlib,json,struct,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_replay_cache_labels_v1 import full_replay_labels
A=Path(r'E:\gpu_training_scratchpad\analysis\ASR18_memory_diagnosis_v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();f=Path(a.folder);require(f.is_relative_to(A),'E-only evidence');f.mkdir(exist_ok=False)
    proof_path=A/'window_replay_PCM_actual_01/report.json';proof=json.loads(proof_path.read_text(encoding='utf-8'))
    require(proof['passed'] is True and proof['planned_replay_draws']==40,'Independent original40 replay reference required')
    checks=[];results=[]
    for ref in proof['references']:
        pin={k:ref['genuine_teacher_feature_cache'][k] for k in ('path','bytes','sha256')}
        ids,observed=full_replay_labels(ref['source_row'],pin)
        sha=hashlib.sha256(b''.join(struct.pack('<I',v) for v in ids)).hexdigest()
        require(ids==ref['full_target_ids'] and sha==ref['target_ids_u32_sha256'],'Every target must match independent actual CPU Torch reference')
        results.append({'source_id':ref['source_id'],'language':ref['language'],'target_ids_u32_sha256':sha,'full_target_ids':ids,'observed':observed})
    checks.append('All original planned40 replay targets byte-exact to independent actual Torch+PCM proof')
    ref=proof['references'][0];pin={k:ref['genuine_teacher_feature_cache'][k] for k in ('path','bytes','sha256')}
    for name,change in (('Wrong sourceID rejected',{'source_id':'different'}),('Wrong fullU rejected',{'target_tokens':ref['source_row']['target_tokens']+1}),
        ('Boolean U rejected',{'target_tokens':True}),('Wrong feature T rejected',{'encoder_frames':ref['source_row']['encoder_frames']+1})):
        try:full_replay_labels({**ref['source_row'],**change},pin)
        except ValueError:checks.append(name)
        else:raise ValueError('Expected rejection: '+name)
    require(not any(v in sys.modules for v in ('torch','numpy','scipy','transformers','pyarrow','soundfile')),'No numeric/model/audio runtime')
    checks.append('No tensor feature decode/audio/runtime/model/optimizer work')
    write_json(f/'report.json',{'passed':True,'check_count':len(checks),'checks':checks,'independent_reference':record(proof_path),
        'source':record(__file__),'reader':record(S/'bpe_replay_cache_labels_v1.py'),'results':results,'rows':len(results),
        'full_target_ID_equality':True,'actual_audio_read_or_feature_tensor_decoded':False,'actual_optimizer_updates':0,
        'all1914_cache_positive_path_exercised':False,'production_enabled':False})
    print(json.dumps({'passed':True,'checks':len(checks),'actual_original_cache_rows':len(results),'actual_optimizer_updates':0}))
if __name__=='__main__':main()
