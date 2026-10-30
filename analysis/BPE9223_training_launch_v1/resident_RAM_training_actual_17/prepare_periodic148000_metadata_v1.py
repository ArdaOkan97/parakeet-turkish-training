import ast,hashlib,json
from pathlib import Path
A=Path(r'E:\gpu_training_scratchpad\analysis\BPE9223_training_launch_v1\resident_RAM_training_actual_17')
p=A/'seal_periodic144000_and_real_quiescence_v1.py';old=p.read_text(encoding='utf-8');s=old.replace('144000','148000')
s=s.replace('first genuine normal periodic CPU closure','actual normal periodic CPU closure').replace('first_actual_normal_periodic_helper2_phase','actual_normal_periodic_helper2_phase')
s=s.replace("closed['completed_updates']==1220","closed['completed_updates']==4020").replace('pending61','pending261').replace('plan61','plan261')
expected="'TR_change_vs140000':{'previous_errors':208,'current_errors':210,'reference_words':825,'error_change':2,'percentage_point_change':2/825*100}"
replacement="'TR_change_vs144000':{'previous_errors':210,'current_errors':entry['score']['word_errors']['tr'],'reference_words':825,'error_change':entry['score']['word_errors']['tr']-210,'percentage_point_change':(entry['score']['word_errors']['tr']-210)/825*100}"
assert s.count(expected)==1;s=s.replace(expected,replacement)
requirement="require(progress['actual_optimizer_updates']>148200,'Actual post148200 positive training')\nnextCK=B/'checkpoints/update_000148200';nextpersistence=read(nextCK/'persistence_verified.json')\nrequire(nextpersistence['passed']and nextpersistence['optimizer_updates']==148200 and nextpersistence['named_tensors_verified']==854 and nextpersistence['weights_verified']==207 and nextpersistence['Adam_states_verified']==215 and nextpersistence['independent_pre_save_state_sha256']==nextpersistence['saved_state_sha256'],'Next standard148200 full854/207/215 PRE-SAVED persistence')\n"
needle="observation=A/'actual_periodic148000_postpositive_progress.json';"
assert s.count(needle)==1;s=s.replace(needle,requirement+needle)
needle="'prediction_files':predictions,'prediction_counts':counts"
replacement="'standard_next148200_checkpoint':{'pointer':pin(nextCK/'pointer.json'),'persistence':pin(nextCK/'persistence_verified.json'),'optimizer_updates':148200,'full854_207_215_PRE_SAVED_passed':True},'prediction_files':predictions,'prediction_counts':counts"
assert s.count(needle)==1;s=s.replace(needle,replacement)
assert 'first_actual_normal'not in s and 'plan61'not in s and "completed_updates']==4020"in s
ast.parse(s);out=A/'seal_periodic148000_and_quiescence_saved148200_v1.py';assert not out.exists();out.write_text(s,encoding='utf-8')
b=out.read_bytes();print(json.dumps({'source':{'path':str(out),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()},'executed':False,'required_actual_progress':'>148200','no_training_action':True}))