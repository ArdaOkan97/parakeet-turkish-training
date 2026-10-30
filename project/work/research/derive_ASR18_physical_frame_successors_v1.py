"""NEW unadmitted shape-bound successors; preserve all executed predecessors."""
import ast,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import require
def save(path,text):
    ast.parse(text);require(not path.exists(),'Preserve existing source successor');path.write_text(text,encoding='utf-8',newline='\n')
source=(S/'bpe_final_DATA_window_planner_v2.py').read_text(encoding='utf-8')
source=source.replace('from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks','from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks\nfrom asr18_physical_feature_geometry_v1 import from_samples,from_encoder_upper_bound')
source=source.replace("'runtime_full_audio_geometry_and_heldout_check_required':True}","'runtime_full_audio_geometry_and_heldout_check_required':True,\n                'declared_frontend_geometry':from_samples(declared['clean16k_samples'])}")
source=source.replace("frames=row['declared_geometry']['mel_frames'] if row['language']=='tr' else 8*t\n        require(type(frames) is int and frames>0 and (frames+7)//8==t,'Conservative full frontend frame bound required')",
    "if row['language']=='tr':\n            shape=from_samples(row['declared_geometry']['clean16k_samples'])\n            require(shape['valid_mask_frames']==row['declared_geometry']['mel_frames'] and shape['encoder_frames']==t,'Exact declared valid frames differ')\n            frames=shape['physical_feature_frames']\n        else:frames=from_encoder_upper_bound(t)['physical_feature_frames_upper_bound']\n        require(type(frames) is int and frames>0,'Conservative physical frontend frame bound required')")
save(S/'bpe_final_DATA_window_planner_v3.py',source)
forecast=(S/'asr18_calibrated_geometry_forecast_v4.py').read_text(encoding='utf-8').replace("'feature_frames':8*t,","'feature_frames':8*t+1,")
forecast=forecast.replace('native_forecast_v4','native_forecast_v5').replace('high_water_v4','high_water_v5')
save(S/'asr18_calibrated_geometry_forecast_v5.py',forecast)
worker=(S/'asr18_precomputed_worker_forecast_v1.py').read_text(encoding='utf-8').replace('native_forecast_v4','native_forecast_v5').replace('worker_phase_forecast_v1','worker_phase_forecast_v2')
save(S/'asr18_precomputed_worker_forecast_v2.py',worker)
test=(R/'validate_ASR18_final_DATA_window_planner_v3.py').read_text(encoding='utf-8').replace('bpe_final_DATA_window_planner_v2','bpe_final_DATA_window_planner_v3')
test=test.replace('from bpe_training_schedule_v1 import WEAK','from bpe_training_schedule_v1 import WEAK\nfrom asr18_physical_feature_geometry_v1 import from_samples,from_encoder_upper_bound')
test=test.replace("checks=[]", "checks=[]\n    require(from_samples(16000)['valid_mask_frames']==100 and from_samples(16000)['physical_feature_frames']==101 and from_samples(16000)['encoder_frames']==13,'Exact1s original frontend geometry')\n    checks.append('N16000 has100 valid mask frames and101 physical feature frames,T13')\n    for n,t in ((1280,1),(2560,2),(1439,1)):\n        shape=from_samples(n);bound=from_encoder_upper_bound(t)\n        require(shape['encoder_frames']==t and shape['physical_feature_frames']<=bound['physical_feature_frames_upper_bound'],'Divisible8 boundary upper bound')\n    checks.append('Multiple8 valid-frame boundary keeps physical8T+1 trailing frame; no crop')")
save(R/'validate_ASR18_final_DATA_window_planner_v4.py',test)
test=(R/'validate_ASR18_geometry_forecast_v4.py').read_text(encoding='utf-8').replace('asr18_calibrated_geometry_forecast_v4','asr18_calibrated_geometry_forecast_v5').replace("[1,2608,128]","[1,2609,128]")
save(R/'validate_ASR18_geometry_forecast_v5.py',test)
