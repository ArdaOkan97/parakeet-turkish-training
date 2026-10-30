import ast,hashlib,json
from pathlib import Path
A=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1\resident_RAM_training_actual_17')
old=(A/'seal_periodic148000_and_quiescence_saved148200_v1.py').read_text(encoding='utf-8')
s=old.replace('148000','152000').replace('148200','152200').replace('pending261','pending461').replace('plan261','plan461')
s=s.replace(s.splitlines()[0],'"""Compact actual152k quality and normal periodic CPU closure."""',1)
expected="'TR_change_vs144000':{'previous_errors':210,'current_errors':entry['score']['word_errors']['tr'],'reference_words':825,'error_change':entry['score']['word_errors']['tr']-210,'percentage_point_change':(entry['score']['word_errors']['tr']-210)/825*100}"
replacement="'TR_change_vs148000':{'previous_errors':208,'current_errors':entry['score']['word_errors']['tr'],'reference_words':825,'error_change':entry['score']['word_errors']['tr']-208,'percentage_point_change':(entry['score']['word_errors']['tr']-208)/825*100}"
assert s.count(expected)==1;s=s.replace(expected,replacement)
assert "closed['completed_updates']==4020"in s and 'plan261'not in s and 'pending261'not in s
assert "progress['actual_optimizer_updates']>152200"in s and 'update_000152200'in s
ast.parse(s);out=A/'seal_periodic152000_and_quiescence_saved152200_v1.py';assert not out.exists();out.write_text(s,encoding='utf-8')
b=out.read_bytes();print(json.dumps({'source':{'path':str(out),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()},'executed':False,'required_actual_progress':'>152200','no_training_action':True}))