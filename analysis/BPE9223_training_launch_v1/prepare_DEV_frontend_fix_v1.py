import ast,copy,difflib,hashlib,json,sys,time
from pathlib import Path
P=Path(r'C:/Users/ardao/Documents/Codex/2026-10-10/i-x20');S=P/'outputs/parakeet_turkish';R=P/'work/research'
N=Path(r'E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1');B=Path(r'E:/gpu_training_scratchpad/runs/BPE9223_resident_RAM_20261018_01')
sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
from bpe_broad_audio_v1 import record,require
from fastmss_owned_process_closure_v2 import dead
area=N/'resident_RAM_DEV_frontend_failure_actual_v1';area.mkdir(exist_ok=True)
inv=B/'invocations/owner_000004';child=read_json(N/'resident_RAM_training_actual_03/current_child.json')
failure=read_json(inv/'actual_failure.json');latest=read_json(B/'latest_checkpoint.json');pointer=read_json(latest['pointer']['path'])
CPU=read_json(inv/'CPU/actual_CPU_ready.json')['producer_identity']
identities=[child['launcher'],failure['producer'],CPU]
require(all(dead(v)for v in identities),'All actual source3 native/CPU identities dead')
journal=record(pointer['journal_prefix']['path'])
require(failure['actual_optimizer_updates']==failure['last_checkpoint']==pointer['committed_updates']==4000
    and latest['full_state_verified'] is True and journal=={'path':pointer['journal_prefix']['path'],'bytes':pointer['journal_prefix']['offset'],'sha256':pointer['journal_prefix']['sha256']},'Saved4000 whole journal; no unsaved new tail')
require('Audio frontend unavailable' in failure['error'],'Exact frontend composition failure')
seal={'kind':'ASR18_actual_resident_RAM_periodic_DEV_frontend_failure_saved4000_v1','owner':'/root/memory_diagnosis',
    'branch':str(B),'actual_owned_exec_session':57717,'actual_exec_tool_exit_code':1,'actual_exec_tool_chunk':'208bb7',
    'native_Popen_wait_nonzero_observed_by_original_wrapper':True,'native_numeric_exit_code_not_persisted':True,
    'normal_native0_claimed':False,'intentional_stop_claimed':False,'producer':failure['producer'],
    'captured_identities':identities,'all_captured_identities_dead':True,
    'uncaptured_parent_PIDs_observed_absent':{str(v):not __import__('psutil').pid_exists(v)for v in (32476,38048)},
    'original_current_child':record(N/'resident_RAM_training_actual_03/current_child.json'),
    'configuration':record(inv/'configuration.json'),'source_configuration':record(N/'resident_RAM_periodic4000_source_v1/configuration.json'),
    'driver_source':record(R/'run_ASR18_BPE9223_resident_RAM_training_v3.py'),
    'original_failure':record(inv/'actual_failure.json'),'original_monitor_completion':record(inv/'monitor_completion.json'),
    'original_native_log':record(N/'resident_RAM_training_actual_03/native_000004.log'),
    'original_CPU_stderr':record(inv/'CPU/CPU_stderr.log'),'original_CPU_ready':record(inv/'CPU/actual_CPU_ready.json'),
    'checkpoint_pointer':latest['pointer'],'checkpoint_persistence':latest['persistence'],'latest_checkpoint':record(B/'latest_checkpoint.json'),
    'saved_optimizer_updates':4000,'actual_optimizer_updates_before_failure':4000,'saved_journal_prefix':pointer['journal_prefix'],
    'whole_journal':journal,'unsaved_tail_updates':[],
    'failed_DEV_directory':str(B/'periodic_DEV/update_000004000'),
    'failed_results':{p.name:record(p)for p in Path(r'E:/gpu_training_scratchpad/results/BPE9223_resident_periodic_DEV_4000').iterdir()if p.is_file()},
    'actual_DEV_predictions_completed':0,'actual_WER_report_produced':False,'periodic4000_still_required_before4001':True,
    'failure_error':failure['error'],'monitor':read_json(inv/'monitor_completion.json'),
    'original_checkpoints_journals_results_and_old_tail_preserved':True}
sealpath=area/'actual_failure_closure.json';require(not sealpath.exists(),'New truthful failure seal');write_json(sealpath,seal)

old=(S/'asr18_resident_serial350_DEV_v1.py').read_text()
compose='''def compose_DEV_processor(training_processor,*,guard):
    """Original local audio frontend plus the exact live migrated tokenizer."""
    from transformers import AutoProcessor
    guard();base=Path(__file__).resolve().parents[2]/'work/base_model_path.txt'
    original=AutoProcessor.from_pretrained(base.read_text().strip(),local_files_only=True)
    guard();tokenizer=training_processor.tokenizer
    require(len(tokenizer)==9223 and training_processor.blank_token_id==8192
        and original.blank_token_id==8192,'Exact original blank and live9223 tokenizer')
    frontend=deepcopy(original.feature_extractor.to_dict())
    original.tokenizer=tokenizer
    require(original.tokenizer is tokenizer and original.blank_token_id==8192
        and original.feature_extractor.to_dict()==frontend,'No frontend equations or training carrier mutation')
    return original,{'kind':'ASR18_actual_original_audio_frontend_with_live9223_tokenizer_v1',
        'processor_class':type(original).__name__,'feature_extractor_class':type(original.feature_extractor).__name__,
        'feature_extractor_configuration_sha256':digest(frontend),'base_model_descriptor':record(base),
        'tokenizer_class':type(tokenizer).__name__,'vocabulary_size':len(tokenizer),'blank_token_id':original.blank_token_id,
        'live_tokenizer_object_reused':True,'training_carrier_replaced':False,'generation_or_decode_policy_changed':False}


'''
new=old.replace('def evaluate(model,processor,optimizer,*,torch,config,pointer,folder,guard):',compose+'def evaluate(model,processor,optimizer,*,torch,config,pointer,folder,guard):',1)
new=new.replace("tag='BPE9223_resident_'+folder.parent.name+'_'+str(saved_updates)","tag='BPE9223_resident_'+folder.parent.name+'_'+str(saved_updates)+'_'+digest(str(folder.resolve()))[:16]",1)
new=new.replace("original=benchmark.inputs_for", "training_tokenizer=processor.tokenizer;training_vocab=digest(training_tokenizer.get_vocab())\n    original=benchmark.inputs_for",1)
new=new.replace('with torch.inference_mode():scores=trainer.evaluate(model,processor,tag)',"DEV_processor,composition=compose_DEV_processor(processor,guard=guard)\n        with torch.inference_mode():scores=trainer.evaluate(model,DEV_processor,tag)",1)
new=new.replace('for module,training in modes:module.training=training',"for module,training in modes:module.training=training\n        require(processor.tokenizer is training_tokenizer and digest(processor.tokenizer.get_vocab())==training_vocab,\n            'Training token carrier and all9223 IDs remain unchanged')",1)
new=new.replace("'source':record(__file__),'TEST_used':False", "'source':record(__file__),'DEV_processor_composition':composition,'TEST_used':False",1)
require(new!=old,'Explicit helper change');ast.parse(new)
target=S/'asr18_resident_serial350_DEV_v2.py';require(not target.exists(),'New inactive helper2');target.write_text(new,encoding='utf-8')
src=N/'resident_RAM_DEV_frontend_source_v1';src.mkdir(exist_ok=True)
(src/'helper1_to2.diff.txt').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='DEV_v1',tofile='DEV_v2')),encoding='utf-8')
print(json.dumps({'failure':record(sealpath),'helper2':record(target)}))
