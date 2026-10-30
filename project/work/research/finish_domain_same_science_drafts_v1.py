import json
from pathlib import Path
HERE=Path(__file__).resolve().parents[2]/'outputs/parakeet_turkish'
path=HERE/'launch_domain30_same_science_resume_v1.py'
text=path.read_text(encoding='utf-8')
start=text.index('PINS=');end=text.index('\n\n\ndef require',start)
pins=json.loads((HERE/'domain30_runtime_chain_pins_v1.json').read_text(encoding='utf-8'))['source_sha256']
text=text[:start]+'PINS='+repr(pins)+text[end:]
path.write_text(text.replace('â€“','-'),encoding='utf-8')
path=HERE/'resume_domain30_same_science_v1.py';text=path.read_text(encoding='utf-8')
start=text.index('    require(0<args.window_hours<=6');end=text.index('\n    source_loop=',start)
text=text[:start]+"""    require(math.isfinite(args.window_hours) and .25<=args.window_hours<=3
        and math.isfinite(args.poll_seconds) and 5<=args.poll_seconds<=15,
        'Bounded0.25-3h / fresh5-15s polling required')
    require(args.run_id and all(c.isalnum() or c in '_-' for c in args.run_id),'Safe SAME run ID required')"""+text[end:]
path.write_text(text.replace('â€“','-'),encoding='utf-8')
