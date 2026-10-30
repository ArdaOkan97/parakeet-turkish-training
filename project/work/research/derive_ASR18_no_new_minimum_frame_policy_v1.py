"""Keep shape arithmetic separate from actual finite normalization validation."""
import ast,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import require
source=(S/'asr18_physical_feature_geometry_v1.py').read_text(encoding='utf-8').replace("samples>=320,'Original normalized frontend needs at least2 valid frames'","samples>=160,'Positive original declared valid-mask geometry required'")
source=source.replace("'encoder_frames':(valid+7)//8,'original_frontend_source_SHA256'", "'encoder_frames':(valid+7)//8,'actual_finite_normalization_validation_still_required':True,'original_frontend_source_SHA256'")
for path,text in ((S/'asr18_physical_feature_geometry_v2.py',source),(S/'bpe_final_DATA_window_planner_v4.py',(S/'bpe_final_DATA_window_planner_v3.py').read_text(encoding='utf-8').replace('asr18_physical_feature_geometry_v1','asr18_physical_feature_geometry_v2'))):
    ast.parse(text);require(not path.exists(),'Preserve source successor');path.write_text(text,encoding='utf-8',newline='\n')
test=(R/'validate_ASR18_final_DATA_window_planner_v4.py').read_text(encoding='utf-8').replace('bpe_final_DATA_window_planner_v3','bpe_final_DATA_window_planner_v4').replace('asr18_physical_feature_geometry_v1','asr18_physical_feature_geometry_v2')
test=test.replace("for n,t in ((1280,1),(2560,2),(1439,1)):","for n,t in ((160,1),(319,1),(1280,1),(2560,2),(1439,1)):")
path=R/'validate_ASR18_final_DATA_window_planner_v5.py';ast.parse(test);require(not path.exists(),'Preserve shape fixture successor');path.write_text(test,encoding='utf-8',newline='\n')
