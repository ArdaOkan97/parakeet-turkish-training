import os,json,sys,hashlib
from pathlib import Path
P=Path(r'C:\Users\ardao\Documents\Codex\2026-10-10\i-x20');R=Path(__file__).parent
sys.path.insert(0,str(P/'outputs/parakeet_turkish'))
from paired_windows_commit_telemetry_v1 import paired_sample
samples=[paired_sample(),paired_sample()]
with (R/'cards_admission.json').open('x') as f:json.dump(samples,f);f.flush();os.fsync(f.fileno())
assert all(s['global_memory_status_ex']['ullAvailPageFile_bytes']>=6028384896 and s['get_performance_info_after']['physical_available_bytes']>=3278384896 for s in samples)
from huggingface_hub import hf_hub_download
sources={'base':('nvidia/parakeet-tdt-0.6b-v3','541d1f99c6b0c3cd0b11a95167540bb8edefd82b','model'),'cv':('ysdede/commonvoice_17_tr_fixed','9d8a025f1ae7c5a2f5c43298c7c4910b03995a4b','dataset'),'fleurs':('google/fleurs','70bb2e84b976b7e960aa89f1c648e09c59f894dd','dataset'),'tsc':('issai/Turkish_Speech_Corpus','d79ea01392882322e8f46657cb635236af25b6d4','dataset'),'worldspeech':('disco-eth/WorldSpeech','main','dataset'),'alania':('cloud0day3/alania-speech-captions-tr','8b0fdd976ed9668b64f5ee0696317a8f241ce39c','dataset'),'kiraat':('serdarcaglar/kiraat','f8afe95abedc19ec6392c77cf470f5d3dcd2788f','dataset'),'yodas3':('espnet/yodas3','81d802f45a1c1e1f5331260484dacf1fbea81981','dataset')}
(R/'cards_scope.json').write_text(json.dumps(sources,indent=2))
out={}
for name,(repo,rev,kind) in sources.items():
    try:
        p=hf_hub_download(repo_id=repo,filename='README.md',revision=rev,repo_type=kind,cache_dir=r'E:\gpu_training_scratchpad\cache\huggingface\hub')
        raw=Path(p).read_bytes();assert len(raw)<2097152
        with (R/(name+'_README_RAW.md')).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        text=raw.decode('utf-8');lines=text.splitlines()
        out[name]={'repo':repo,'revision':rev,'sha256':hashlib.sha256(raw).hexdigest(),'license_lines':[l for l in lines if any(x in l.lower() for x in ('licen','cc-by','cc0','cc by','non-commercial'))][:15]}
    except Exception as e:out[name]={'repo':repo,'error_type':type(e).__name__}
(R/'model_dataset_cards.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False))
