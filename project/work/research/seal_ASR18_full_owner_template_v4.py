"""Light disabled-template join for source-reviewed builder6 raw-state assertion."""
from pathlib import Path
import ast
P=Path(__file__).resolve().parents[2];source=P/'work/research/seal_ASR18_full_owner_template_v1.py'
text=source.read_text(encoding='utf-8-sig').replace('full_owner_exact_ROOT_review_inputs_v4.json','full_owner_exact_ROOT_review_inputs_v8.json')
text=text.replace("R/'run_ASR18_full_run_owner_v9.py'","R/'run_ASR18_full_run_owner_v11.py'")
ast.parse(text);exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
