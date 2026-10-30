"""NEW policy reports the actual incremental admission ceiling explicitly."""
import ast,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';R=P/'work/research';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import require
source=(S/'asr18_calibrated_geometry_forecast_v3.py').read_text(encoding='utf-8')
source=source.replace("value=super().budget(rows=rows,current_private_bytes=current_private_bytes,update_in_process=update_in_process)",
    "require(update_in_process!=1 or current_private_bytes<=3_000_000_000,'First native private observation exceeds reviewed3GB startup envelope; stop before native work')\n        value=super().budget(rows=rows,current_private_bytes=current_private_bytes,update_in_process=update_in_process)\n        value['calibrated_target_private_bytes']=value['absolute_private_forecast_bytes']\n        value['absolute_private_forecast_bytes']=max(value['absolute_private_forecast_bytes'],current_private_bytes+value['incremental_bytes'])\n        value['first_native_startup_private_upper_bound_bytes']=3_000_000_000")
source=source.replace('ASR18_finite_measured_geometry_native_forecast_v3','ASR18_finite_measured_geometry_native_forecast_v4').replace('ASR18_finite_native_forecast_high_water_v3','ASR18_finite_native_forecast_high_water_v4')
ast.parse(source);target=S/'asr18_calibrated_geometry_forecast_v4.py';require(not target.exists(),'Preserve policy successor');target.write_text(source,encoding='utf-8',newline='\n')
test=(R/'validate_ASR18_geometry_forecast_v3.py').read_text(encoding='utf-8').replace('asr18_calibrated_geometry_forecast_v3','asr18_calibrated_geometry_forecast_v4')
test=test.replace("require(not any(n in sys.modules", "reject('First native observation outside3GB startup envelope stops',lambda:policy.budget(rows=scope,current_private_bytes=3_000_000_001,update_in_process=1))\n    elevated=policy.budget(rows=scope,current_private_bytes=6_000_000_000,update_in_process=2)\n    require(elevated['absolute_private_forecast_bytes']>=6_250_000_000,'Reported absolute ceiling cannot understate current plus required increment')\n    checks.append('Later elevated residency reports full actual incremental admission ceiling')\n    require(not any(n in sys.modules")
ast.parse(test);out=R/'validate_ASR18_geometry_forecast_v4.py';require(not out.exists(),'Preserve fixture successor');out.write_text(test,encoding='utf-8',newline='\n')
