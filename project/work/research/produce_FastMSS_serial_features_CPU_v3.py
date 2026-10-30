"""One serial update using independent clean PCM, exact original CPU frontend."""
import argparse,gc,inspect,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
import fastmss_row_serial_cpu_v3 as core

def main(request_path,output):
    core.environment();guard=core.resource_guard
    request,plan,update,contract,validated=core.load_request(core.record(request_path,guard),guard)
    guard=core.stage_guard(contract,'frontend')
    admission=core.admit(contract['resource_review']['frontend_incremental_bytes'],guard)
    np,torch=core.numpy_runtime(frontend=True)
    import scipy
    identity=core.bridge.frozen_augmentation(plan['augmentation_identity'],guard)
    core.require(np.__version__==identity['numpy_version'] and scipy.__version__==identity['scipy_version'],'Exact frozen augmentation numerical runtime; never rewrite identity')
    from transformers import AutoFeatureExtractor
    declared=contract['frontend'];extractor=AutoFeatureExtractor.from_pretrained(declared['base_path'],local_files_only=True)
    core.require(type(extractor).__name__=='ParakeetFeatureExtractor' and core.same(core.record(inspect.getfile(type(extractor)),guard),declared['source']),'Original source-bound Parakeet frontend, no model load')
    weak_transform=None;binding=None
    if update['phase']==core.bridge.WEAK:
        import fastmss_train_augmentation_v1 as adapter
        from fastmss_ordered_draw_binding_v1 import FastMSSOrderedDrawBinding,AugmentationDrawContext
        augmenter=adapter.TrainWaveformAugmenter(adapter.AugmentationPolicy.from_json(identity['policy_file']['path']),guard=guard)
        binding=FastMSSOrderedDrawBinding(augmenter,policy_path=identity['policy_file']['path'],adapter_path=adapter.__file__)
        core.require(core.same(binding.identity,identity),'Actual frozen policy/registry/NumPy/SciPy/source identity must match exactly')
        def weak_transform(wave,row,context):
            return reader.transform(lambda w,r,c:augmenter.apply(w,r,**AugmentationDrawContext(**c).adapter_context()),wave,row,context)
    reader=core.OriginalRows(validated,raw_contract=contract['raw_lifetime_review'],
        private_limit_bytes=contract['resource_review']['frontend_maximum_private_bytes'],frontend_enabled=True,guard=guard);cursor=0;actual_geometry=[]
    refs=core.read_bound(core.read_bound(request['input_binding'],guard)['independent_clean_PCM'],guard)['references']
    def clean_reader(row):
        nonlocal cursor
        i=cursor;core.require(i<len(update['rows']) and core.same(row['audio_draw_context'],update['rows'][i]['context']),'Exact serial original row context')
        wave=reader(row,update['rows'][i],update['phase'])
        independent=refs[i]
        core.require(core.same(reader.last,independent['actual_geometry']) and core.same(reader.geometry_proof,independent['before_draw_geometry_proof']) and core.same(reader.original_source_files,independent['original_source_files']),'Independent source/full native/rate/sample/T/PCM proof must reproduce before features')
        if update['rows'][i]['role']=='replay':core.require(core.same(reader.replay_proof,independent['genuine_teacher_feature_cache']),'Genuine original replay cache proof differs')
        cursor+=1;return wave
    def original_frontend(wave):
        guard();inputs=dict(extractor([wave],**core.sink.FRONTEND_ARGS))
        core.require(set(inputs)=={'input_features','attention_mask'},'Complete original frontend outputs')
        mask=inputs['attention_mask'].bool()
        features=inputs['input_features'].masked_fill(~mask.unsqueeze(-1),0).detach().cpu().contiguous().numpy()
        arrays={'input_features':features,'attention_mask':mask.detach().cpu().contiguous().numpy()}
        actual_geometry.append(core.check_frontend(arrays,len(wave),np));return arrays
    def frontend(wave):return reader.frontend(original_frontend,wave)
    try:
        producer=core.sink.PhaseAwareUpdateSink(request['owner'],request['update_index'],request['input_binding'],
            prior_closures=request['prior_closures'],clean_reader=clean_reader,weak_transform=weak_transform,
            transform_identity=identity,frontend=frontend,guard=guard)
        for i in range(len(update['rows'])):producer.prepare_row(i);gc.collect();reader.finish_raw_row()
        update_record=producer.finish()
    finally:reader.close()
    core.require(cursor==len(update['rows']) and len(actual_geometry)==cursor,'Exactly once full ordered row processing')
    core.require(not any(k=='pyarrow' or k.startswith('pyarrow.') or k.startswith('nemo.collections.asr.models') for k in sys.modules),'No Arrow/ASR model in frontend child')
    result={'kind':'FastMSS_actual_serial_CPU_frontend_update_v1','passed':True,'producer':core.identity(),
        'worker_source':core.record(__file__,guard),'parent_source':contract['sources']['controller'],
        'owner':request['owner'],'update_record':update_record,'request':core.record(request_path,guard),
        'bridge':request['bridge'],'update_index':request['update_index'],'phase':update['phase'],
        'actual_geometry':actual_geometry,'frontend':declared,'Torch_file':str(Path(torch.__file__).resolve()),
        'NumPy':np.__version__,'SciPy':scipy.__version__,'Torch':torch.__version__,'torch_threads':torch.get_num_threads(),
        'augmentation_identity':identity,'actual_optimizer_updates':0,'sampler_journal_frontier_advanced':False,
        'raw_PCM_files_written':0,'simultaneous_raw_jobs':1,'CPU_stage_admission':admission,
        'cleaner_augmenter_constructed':False if update['phase']==core.bridge.CLEANER else None,
        'actual_child_exit_observed_in_child':False}
    result['raw_lifetime_receipt']=reader.raw_receipt()
    core.write_new(core.scoped(output),result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.request,a.output)
