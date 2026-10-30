import ast,difflib,hashlib,json
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');S=P/'outputs/parakeet_turkish';N=Path(__file__).parent
old=S/'fastmss_resident_feature_stream_v1.py';new=S/'fastmss_resident_feature_stream_v2.py'
s=old.read_text();a="    reader = core.OriginalRows(validated, raw_contract=contract['raw_lifetime_review'],";b="    from fastmss_original_clip_geometry_overlay_v1 import OriginalRows\n    reader = OriginalRows(validated, geometry_overlay=config['original_clip_geometry_overlay'], raw_contract=contract['raw_lifetime_review'],"
assert s.count(a)==1;s=s.replace(a,b)
a="        provenance = {'protocol': pin(__file__), 'numerical_singlepass_source': pin(single.__file__),";b="        provenance = {'protocol': pin(__file__), 'original_clip_geometry_overlay': bound(configuration_path)['original_clip_geometry_overlay'],\n                      'numerical_singlepass_source': pin(single.__file__),"
assert s.count(a)==1;s=s.replace(a,b);assert not new.exists();new.write_text(s,encoding='utf-8')
for path in (new,S/'fastmss_original_clip_geometry_overlay_v1.py'):ast.parse(path.read_text())
(N/'resident_stream1_to2.diff.txt').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),s.splitlines(True),fromfile=str(old),tofile=str(new))),encoding='utf-8')
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
h={'kind':'ASR18_inactive_one_original_full_clip_geometry_source_handoff_v1','overlay':pin(S/'fastmss_original_clip_geometry_overlay_v1.py'),'CPU_stream':pin(new),'CPU_delta':pin(N/'resident_stream1_to2.diff.txt'),'certificate':pin(N/'one_original_full_clip_geometry_certificate_v1.json'),'observation':pin(N/'full_member_actual_02/report.json'),'scope':'Inactive only; pending ROOT source/CPU-control acceptance before any admission','configuration_policy_schema':{'original_clip_geometry_overlay':{'source':'overlay record3','certificate':'certificate record3','ROOT_review':'ROOT source/actual CPU review record3'}}}
(N/'one_row_overlay_source_handoff_v1.json').write_text(json.dumps(h,indent=2)+'\n',encoding='utf-8');print(json.dumps(h))
