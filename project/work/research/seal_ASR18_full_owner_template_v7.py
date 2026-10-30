"""Light disabled template join for reviewed normal-resume genesis source."""
from pathlib import Path
import ast
P=Path(__file__).resolve().parents[2];source=P/'work/research/seal_ASR18_full_owner_template_v1.py'
text=source.read_text(encoding='utf-8-sig')
text=text.replace('full_owner_exact_ROOT_review_inputs_v4.json','full_owner_exact_ROOT_review_inputs_v12.json')
text=text.replace("R/'run_ASR18_full_run_owner_v9.py'","R/'run_ASR18_full_run_owner_v15.py'")
text=text.replace('planned_native_updates=401072,maximum_owner_validation_updates=8,maximum_validation_updates_per_window=4,',
    'planned_native_updates=401072,maximum_owner_validation_updates=8,maximum_validation_updates_per_window=4,\n        normal_boundary_resume_enabled=True,owner_validation_pause_after_updates=4,')
ast.parse(text);exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
