"""Correct fixture-owned SQLite closure; preserve prior executed controls."""
import ast,sys
from pathlib import Path
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import require
source=(P/'work/research/validate_ASR18_final_DATA_window_planner_v2.py').read_text(encoding='utf-8')
source=source.replace('import argparse,copy,hashlib,json,sqlite3,struct,sys','import argparse,copy,hashlib,json,sqlite3,struct,sys\nfrom contextlib import contextmanager')
source=source.replace('\ndef main():','\n@contextmanager\ndef sqlite_owned(path):\n    connection=sqlite3.connect(path)\n    try:\n        with connection:yield connection\n    finally:connection.close()\n\ndef main():')
source=source.replace('with sqlite3.connect(', 'with sqlite_owned(')
ast.parse(source);target=P/'work/research/validate_ASR18_final_DATA_window_planner_v3.py'
require(not target.exists(),'Preserve executed fixture sources');target.write_text(source,encoding='utf-8',newline='\n')
