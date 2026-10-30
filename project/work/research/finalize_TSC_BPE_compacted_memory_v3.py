"""Pin only unactivated compacted memoryv3; no corpus/ML/state/GPU action."""
from pathlib import Path
import ast
import hashlib
import re
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'outputs/parakeet_turkish'
worker=HERE/'train_lora_batched_tsc_bpe_row_schedule_memory_v3.py'
helper=HERE/'tsc_bpe_row_schedule_memory_v3.py'
controller=HERE/'run_TSC_bpe_row_schedule_memory_v3.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
tree=ast.parse(worker.read_text(encoding='utf-8'))
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
sync=[n.lineno for n in ast.walk(run) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize'];assert len(sync)==1
c=controller.read_text(encoding='utf-8')
c=re.sub(r"WORKER_SHA = '[0-9a-f]+'","WORKER_SHA = '"+sha(worker)+"'",c)
c=re.sub(r"TRANSITION_SHA = '[0-9a-f]+'","TRANSITION_SHA = '"+sha(helper)+"'",c)
c=re.sub(r'EXPECTED_SYNC_LINE = [0-9]+','EXPECTED_SYNC_LINE = '+str(sync[0]),c)
ast.parse(c);controller.write_text(c,encoding='utf-8')
print({'helper':sha(helper),'worker':sha(worker),'sync':sync[0],'controller':sha(controller),
 'compactor':sha(HERE/'compact_training_row_objects_v1.py')})
