"""All fixed/published reference PCM bindings. No model or TEST predictions."""
import argparse,hashlib,json,os,sys,wave,gc
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad');AREA=ROOT/'analysis/ASR18_memory_diagnosis_v1'
sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from bpe_broad_audio_v1 import record,require,resample,MAX_AUDIO_COLUMN_UNCOMPRESSED,MAX_NATIVE_PCM_BYTES
from atomic_io_retry_v1 import write_json

def main():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(AREA/'heldout_PCM_binding_v1'));a=p.parse_args();folder=Path(a.output)
    require(folder.resolve().is_relative_to(ROOT.resolve()) and os.environ['CUDA_VISIBLE_DEVICES']=='-1','E-only CPU binding')
    folder.mkdir(parents=True,exist_ok=False)
    metadata_path=ROOT/'analysis/BPE_weak_then_clean_v1/heldout_metadata_bindings_v1.json';metadata=json.loads(metadata_path.read_text())
    references={};fixed={};sources=[];count=0;journal=folder/'reference_PCM_rows.jsonl'
    def append(row):
        nonlocal count
        references.setdefault(row['PCM_i16_sha256'],[]).append(row['reference_id']);count+=1
        with journal.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        if count%100==0:write_json(folder/'progress.json',{'reference_clips_bound':count,'unique_PCM_hashes':len(references),'last_reference_id':row['reference_id']})
    for binding in metadata['files']:
        path=Path(binding['path']).resolve()
        require(path.is_relative_to(PROJECT) or path.is_relative_to(ROOT.resolve()),'Only existing project/E heldout sources')
        if path.suffix!='.jsonl':continue
        actual=record(path);require(actual['sha256']==binding['sha256'],'Fixed heldout source manifest changed');sources.append(actual)
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            row=json.loads(line);require('prediction' not in row and 'hypothesis' not in row,'Never TEST predictions')
            sid=row['source_id'];audio=Path(row['audio_filepath']).resolve()
            require(audio.is_relative_to(PROJECT) or audio.is_relative_to(ROOT.resolve()),'Existing allowed reference audio only')
            if sid in fixed:require(fixed[sid]==str(audio),'Same reference source with conflicting audio');continue
            fixed[sid]=str(audio)
            with sf.SoundFile(audio) as f:
                require(f.samplerate==16000 and f.channels==1,'Existing fixed mono16k reference contract')
                require(f.frames*4<=MAX_NATIVE_PCM_BYTES,'Reference exceeds bounded PCM contract')
                samples=f.read(dtype='float32',always_2d=False)
            canonical=np.clip(np.rint(samples*32768),-32768,32767).astype('<i2')
            append({'reference_id':sid,'role':'fixed_DEV_TEST_source_audio_only','path':str(audio),'samples':len(samples),
                'PCM_i16_sha256':hashlib.sha256(canonical.tobytes()).hexdigest()});del samples,canonical
    require(len(fixed)==1050,'All1050 fixed heldout identities required')
    published=0
    for binding in metadata['files']:
        if binding.get('role')!='KIRAAT_published_heldout':continue
        path=Path(binding['path']);actual=record(path);require(actual['sha256']==binding['sha256'],'Pinned published heldout source changed');sources.append(actual)
        pf=pq.ParquetFile(path)
        for g in range(pf.metadata.num_row_groups):
            group=pf.metadata.row_group(g);size=sum(group.column(c).total_uncompressed_size for c in range(group.num_columns) if group.column(c).path_in_schema.startswith('audio.'))
            require(size<=MAX_AUDIO_COLUMN_UNCOMPRESSED,'Heldout source exceeds bounded Parquet reader; never omit references')
        import io
        for batch in pf.iter_batches(batch_size=1,columns=['audio','id','source_id'],use_threads=False):
            row=batch.to_pylist()[0]
            with sf.SoundFile(io.BytesIO(row['audio']['bytes'])) as f:
                require(f.samplerate==24000 and f.channels==1 and f.frames*4<=MAX_NATIVE_PCM_BYTES,'Full pinned published FLAC contract')
                pcm=f.read(dtype='float32',always_2d=False)
            # Embedded audio is already clipped; no source-timestamp trim.
            pcm=resample(pcm,24000);canonical=np.clip(np.rint(pcm*32768),-32768,32767).astype('<i2')
            append({'reference_id':'kiraat:'+row['id'],'recording_source_id':row['source_id'],'role':'published_DEV_TEST_source_audio_only',
                'path':str(path),'samples':len(pcm),'PCM_i16_sha256':hashlib.sha256(canonical.tobytes()).hexdigest()});published+=1
            del pcm,canonical,row
        del batch,pf;gc.collect();pa.default_memory_pool().release_unused()
    require(published>0 and len([b for b in metadata['files'] if b.get('role')=='KIRAAT_published_heldout'])==4,'All4 published heldout files required')
    report={'passed':True,'complete':True,'kind':'ASR18_all_reference_PCM_binding_v1','fixed_source_ids':len(fixed),'published_KIRAAT_clips':published,
        'reference_clips':count,'PCM_i16_sha256':sorted(references),'hash_to_reference_ids':references,'metadata_binding':record(metadata_path),
        'sources':sources,'PCM_row_journal':record(journal),'builder':record(__file__),'resample_implementation':record(PROJECT/'outputs/parakeet_turkish/bpe_broad_audio_v1.py'),
        'sample_rate':16000,'quantization':'clip(rint(float32*32768),-32768,32767) little-endian_i16',
        'all_reference_PCM_bindings_complete':True,'all_training_audio_prechecked':False,'fuzzy_cross_codec_duplicate_detection_claimed':False,
        'runtime_positive_match_policy':'stop before consumed draw; explicit heldout-exclusion/final-catalog reconciliation required',
        'TEST_predictions_or_model_used':False,'TRAIN_admitted':False}
    require('torch' not in sys.modules,'No model/CUDA runtime imports');write_json(folder/'report.json',report)
    print(json.dumps({'passed':True,'fixed':len(fixed),'published':published,'total':count}),flush=True)

if __name__=='__main__':main()
