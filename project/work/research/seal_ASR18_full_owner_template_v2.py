"""Light configuration join successor for exact native5/runtime6 source pins."""
from pathlib import Path
import ast
P=Path(__file__).resolve().parents[2];source=P/'work/research/seal_ASR18_full_owner_template_v1.py'
text=source.read_text(encoding='utf-8-sig').replace('full_owner_exact_ROOT_review_inputs_v4.json','full_owner_exact_ROOT_review_inputs_v5.json')
ast.parse(text);exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
