"""Original fixed350 evaluator/scorer with the resident optimizer retained idle.

No optimizer step, augmentation or sampler operation occurs. This explicitly
does not claim the old separate optimizer-free worker/process protocol.
"""
from copy import deepcopy
import json
from pathlib import Path
from atomic_io_retry_v1 import write_json
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import digest


def evaluate(model,processor,optimizer,*,torch,config,pointer,folder,guard):
    import numpy as np
    import benchmark
    import train_lora_batched_tsc_bpe_full_decoder_v1 as trainer
    from asr18_full_run_serial_DEV_9223_v3 import _capture_RNG,_restore_RNG
    from asr18_live_state_fingerprint_v1 import live_state_fingerprint,require_exact_live_restore
    from asr18_BPE9223_lifecycle_v3 import REFERENCES,LIMITS,SCORER_SHA,GATE_SHA
    from metrics import score_transcriptions
    policy_pin=config['full_run_identity']['scorer_policy']
    require(record(policy_pin['path'])==policy_pin,'Same original official fixed350 scorer policy')
    policy=json.loads(Path(policy_pin['path']).read_text(encoding='utf-8'))
    require(policy['reference_words']==REFERENCES and policy['retention_limits']==LIMITS
        and record(policy['official_scorer']['path'])==policy['official_scorer']
        and policy['official_scorer']['sha256']==SCORER_SHA
        and record(policy['integer_gate_contract']['path'])==policy['integer_gate_contract']
        and policy['integer_gate_contract']['sha256']==GATE_SHA,'Unchanged official scorer and all6 integer gates')
    folder=Path(folder);require(not folder.exists(),'Fresh actual pass DEV output');folder.mkdir()
    saved_updates=optimizer.updates;before=live_state_fingerprint(optimizer,torch=torch,guard=guard)
    rng=_capture_RNG(torch,np);modes=[(m,m.training)for m in model.modules()]
    tag='BPE9223_resident_'+folder.parent.name+'_'+str(saved_updates)
    result=Path(r'E:\gpu_training_scratchpad')/'results'/tag
    require(not result.exists() and not result.with_name(tag+'_pt').exists(),'No cached predictions or partial DEV adoption')
    original=benchmark.inputs_for
    def inputs(processor,row):guard();return original(processor,row)
    benchmark.inputs_for=inputs
    try:
        with torch.inference_mode():scores=trainer.evaluate(model,processor,tag)
    finally:
        benchmark.inputs_for=original;_restore_RNG(torch,np,rng)
        for module,training in modes:module.training=training
    require(optimizer.updates==saved_updates and not optimizer.poisoned,'No positive Adam or optimizer mutation in DEV')
    require_exact_live_restore(before,live_state_fingerprint(optimizer,torch=torch,guard=guard))
    predictions={lang:record(result/('validation_'+lang+'.jsonl'))for lang in REFERENCES}
    counts={}
    for language,words in REFERENCES.items():
        def lines(pin):
            require(record(pin['path'])==pin,'Exact immutable saved DEV membership/prediction')
            return [json.loads(v)for v in Path(pin['path']).read_text(encoding='utf-8-sig').splitlines()if v.strip()]
        expected,actual=lines(policy['DEV_manifests'][language]),lines(predictions[language])
        require(len(expected)==len(actual)==50 and len({r['source_id']for r in expected})==50
            and [r['source_id']for r in expected]==[r['source_id']for r in actual],'Exact original seven50-clip sets/order')
        for source,row in zip(expected,actual):
            require(source['language']==row['language']==language and row['split']=='validation'
                and source['text']==row['reference'],'Same original language/reference and validation only')
            for key in ('sentence_id','dataset_revision','audio_sha256','wav_sha256'):
                require(source.get(key)==row.get(key),'Unchanged actual fixed DEV source provenance')
        score=score_transcriptions(actual,language=language);score.pop('details')
        require(score['utterances']==50 and score['reference_words']==words
            and score==scores['validation_'+language]['asr'],'Same official saved-text score/counts')
        counts[language]=score['word_errors']
    require(all(not p.name.startswith('test_')for p in result.iterdir()),'TEST remains untouched')
    score={'word_errors':counts,'reference_words':deepcopy(REFERENCES),'retention_limits':deepcopy(LIMITS),
        'strict_all_six_passed':all(counts[k]<=v for k,v in LIMITS.items()),'turkish_wer':counts['tr']/825,
        'full350_DEV_complete':True,'TEST_used':False}
    report={'kind':'ASR18_actual_resident_optimizer_idle_serial350_DEV_v1','passed':True,'checkpoint_pointer':pointer,
        'optimizer_objects_retained_but_not_invoked':True,'separate_optimizer_free_worker_claimed':False,
        'actual_native_updates_in_DEV':0,'all207_threeAdam_all4RNG_unchanged':True,'optimizer_updates':saved_updates,
        'predictions':predictions,'scores':scores,'official_saved_text_score':score,'scorer_policy':policy_pin,
        'source':record(__file__),'TEST_used':False,'retention_early_stop':False}
    path=folder/'report.json';write_json(path,report);return record(path),score
