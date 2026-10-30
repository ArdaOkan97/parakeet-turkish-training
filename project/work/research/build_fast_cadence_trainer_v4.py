"""Narrow new 250-update DEV cadence; all previous files stay immutable."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/parakeet_turkish'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def create(name,text):
    p=D/name
    if p.exists():raise FileExistsError(p)
    ast.parse(text);p.write_text(text,encoding='utf-8',newline='\n');return sha(p)
text=(D/'train_lora_batched_data_v3.py').read_text(encoding='utf-8')
def once(left,right):
    global text
    if text.count(left)!=1:raise ValueError('Unique span required: '+left[:80])
    text=text.replace(left,right)
text=text.replace('Separate public-data-v2','Separate public-data-v3').replace('train_lora_batched_data_v3.py','train_lora_batched_data_v4.py')
once('"schedule","min_lr_ratio","eval_per_epoch","eval_epoch_fraction","min_improvement",','"schedule","min_lr_ratio","eval_per_epoch","eval_epoch_fraction","eval_every_updates","min_improvement",')
once('        if sampler.turkish_epoch + 1e-10 >= next_eval or step == configured_steps:',
     '        if (args.eval_every_updates is not None and step % args.eval_every_updates == 0) or sampler.turkish_epoch + 1e-10 >= next_eval or step == configured_steps:')
once('    p.add_argument("--eval-per-epoch",type=float,default=1.)',
     '    p.add_argument("--eval-per-epoch",type=float,default=1.); p.add_argument("--eval-every-updates",type=int)')
once('    if args.steps is not None and args.steps <= 0:',
     '    if args.eval_every_updates is not None and args.eval_every_updates <= 0:\n        raise ValueError("Development update interval must be positive")\n    if args.steps is not None and args.steps <= 0:')
tree=ast.parse(text)
sync=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize']
if len(sync)!=1 or sync[0].lineno!=539:raise ValueError('Exact reviewed update hook line539 required')
trainer=create('train_lora_batched_data_v4.py',text)
text=(D/'launch_with_cache_reclaim_v3.py').read_text(encoding='utf-8').replace('train_lora_batched_data_v3.py','train_lora_batched_data_v4.py').replace('519858efc3532d1dbde9932c45da562187afdb96ec7bcce9beaf5b229df103b8',trainer).replace('cache_reclamation_training_launcher_v3','cache_reclamation_training_launcher_v4')
cache=create('launch_with_cache_reclaim_v4.py',text)
text=(D/'launch_with_memory_guard_v6.py').read_text(encoding='utf-8').replace('train_lora_batched_data_v3.py','train_lora_batched_data_v4.py').replace('launch_with_cache_reclaim_v3','launch_with_cache_reclaim_v4').replace('519858efc3532d1dbde9932c45da562187afdb96ec7bcce9beaf5b229df103b8',trainer).replace('ea1072c1b59a40b621cd9f40ec69b3420e86988a50a50b4b288b0208e8e5d97f',cache).replace('memory_guard_training_launcher_v6','memory_guard_training_launcher_v7')
wrapper=create('launch_with_memory_guard_v7.py',text)
print(json.dumps(dict(trainer=trainer,cache_launcher=cache,memory_launcher=wrapper)))
