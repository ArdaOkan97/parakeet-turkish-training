"""21 finite listening exports from supplied TRAIN speech and three recorded noises."""
import base64
import hashlib
import html
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import sys

from ASR18_continuous_data_guard_v5 import DataIOGuard, ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO

P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish'
sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
import fastmss_real_noise_v1 as mixer

D=ROOT/'analysis/FastMSS_real_recorded_noise_listening_v1'


def main():
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    guard=DataIOGuard('real_recorded_noise_listening_generator',heavy=False,maximum_bytes=200_000_000);guard.wait();pins=MetadataIO(guard)
    target=D/'recorded-noise-turkish-listening.html';sample_dir=D/'samples';assert not target.exists() and not sample_dir.exists();sample_dir.mkdir()
    validated=mixer.validate_registry(D/'real_noise_registry_v1.json',guard=guard.wait)
    registry=validated['registry'];assets={v['id']:v for v in registry['assets']}
    selection=read_json(D/'selected_TRAIN_utterances_v1.json')
    assert selection['DEV_or_TEST_used'] is False and selection['recognizer_used'] is False and len(selection['clips'])==3
    mixer_pin=pins.binding(S/'fastmss_real_noise_v1.py')
    assert mixer_pin['sha256']=='7a7587027c112533a081e2a825fa670d92b3a6233ec5a5c935069d2baa051cbc'
    noise_arrays={};decode_receipts={}
    for key,asset in assets.items():
        guard.wait();offset_s={'cafe':7,'fan':2,'traffic':25}[key]
        with sf.SoundFile(asset['audio']['path']) as stream:
            sr=stream.samplerate;channels=stream.channels
            assert (sr,channels,stream.frames)==(asset['audio']['native_sample_rate'],asset['audio']['native_channels'],asset['audio']['native_frames'])
            start=int(offset_s*sr);count=min(30*sr,stream.frames-start)
            assert start>=0 and count>=20*sr and count*channels*4<=32*1024**2
            stream.seek(start);native=stream.read(count,dtype='float32',always_2d=True);guard.pace(native.nbytes)
        mono=np.mean(native,axis=1,dtype=np.float32);del native
        divisor=math.gcd(sr,16000)
        noise=np.ascontiguousarray(resample_poly(mono,16000//divisor,sr//divisor,window=('kaiser',5.0),padtype='constant'),dtype=np.float32);del mono
        assert len(noise)==(count*16000+sr-1)//sr and len(noise)<=30*16000 and np.isfinite(noise).all()
        noise_arrays[key]=noise
        decode_receipts[key]=dict(source_asset_SHA256=asset['audio']['sha256'],native_start_frame=start,native_frames=count,native_rate=sr,native_channels=channels,
            downmix='float32 arithmetic mean',resampler='scipy.signal.resample_poly, Kaiser5 constant padding',resampled_rate=16000,resampled_frames=len(noise),
            resampled_f32_SHA256=hashlib.sha256(memoryview(noise).cast('B')).hexdigest(),persistent_PCM_file_created=False)
    samples=[];utterances=[];audio_payloads={}
    def rms(array):return math.sqrt(float(np.dot(array.astype(np.float64),array.astype(np.float64)))/len(array))
    def save(key,wave,metadata,clean_component=None):
        guard.wait();assert wave.dtype==np.float32 and wave.ndim==1 and np.isfinite(wave).all() and float(np.abs(wave).max())<=.950001
        buffer=io.BytesIO();sf.write(buffer,wave,16000,format='FLAC',subtype='PCM_16');encoded=buffer.getvalue();assert len(encoded)<2_000_000
        decoded,sr=sf.read(io.BytesIO(encoded),dtype='float32');assert sr==16000 and decoded.ndim==1 and len(decoded)==len(wave) and np.isfinite(decoded).all()
        peak=float(np.abs(decoded).max());assert peak<1 and int(np.count_nonzero(np.abs(decoded)>=1))==0
        if clean_component is not None:
            measured=20*math.log10(rms(clean_component)/rms(decoded-clean_component));assert abs(measured-metadata['snr_db'])<.02
            metadata['post_FLAC_PCM16_measured_clip_RMS_SNR_dB']=measured
        path=sample_dir/(key+'.flac');path.write_bytes(encoded);guard.pace(len(encoded))
        result=dict(id=key,**metadata,audio=pins.binding(path),sample_rate=16000,samples=len(wave),duration_s=len(wave)/16000,
            decoded_peak=peak,decoded_clipped_samples=0,encoded_format='FLAC / PCM16, preview-only quantization',
            duration_exact=True,decoded_all_finite=True)
        samples.append(result);audio_payloads[key]=base64.b64encode(encoded).decode('ascii')
    for clip in selection['clips']:
        row=clip['source_row'];path=Path(row['audio_filepath']).resolve()
        assert path.is_relative_to(P) and row['dataset_split']=='train' and row['language']=='tr'
        source_pin=pins.binding(path,maximum=32_000_000)
        assert source_pin['sha256']==row['wav_sha256']
        with sf.SoundFile(path) as stream:
            assert stream.samplerate==16000 and stream.channels==1 and 8*16000<=stream.frames<=20*16000
            assert ((stream.frames//160)+7)//8==row['encoder_frames']
            wave=stream.read(dtype='float32');guard.pace(wave.nbytes)
        wave=np.ascontiguousarray(wave,dtype=np.float32);assert np.isfinite(wave).all()
        segments={};segment_receipts={};peaks=[]
        for key in clip['noise_conditions']:
            seed=int.from_bytes(hashlib.sha256((clip['id']+'\0'+key+'\0real-noise-v1').encode()).digest()[:8],'big')
            segment,receipt=mixer.prepare_noise_segment(noise_arrays[key],len(wave),seed,guard=guard.wait,source_offset_samples=int(decode_receipts[key]['native_start_frame']*16000/decode_receipts[key]['native_rate']))
            assert receipt['wrap_seams']==0
            segments[key]=segment;segment_receipts[key]=receipt
            for snr in (20,10,5):
                prepared=mixer.prepare_mix(wave,segment,snr,guard=guard.wait);peaks.append(prepared['receipt']['unscaled_peak']);del prepared
        gain=mixer.common_gain_for_conditions(wave,peaks,guard=guard.wait,ceiling=.95)
        clean=(wave*gain).astype(np.float32)
        save(clip['id']+'_clean',clean,dict(utterance_id=clip['id'],environment='clean',snr_db=None,common_utterance_gain=gain,transcript_unchanged=True))
        for key in clip['noise_conditions']:
            for snr in (20,10,5):
                prepared=mixer.prepare_mix(wave,segments[key],snr,guard=guard.wait);rendered=mixer.render_mix(prepared,gain,guard=guard.wait)
                save(clip['id']+'_'+key+'_'+str(snr),rendered['audio'],dict(utterance_id=clip['id'],environment=key,snr_db=snr,
                    registry_binding=validated['binding'],source_asset_SHA256=assets[key]['audio']['sha256'],segment=segment_receipts[key],noise_decode=decode_receipts[key],
                    mix_receipt=rendered['receipt'],common_utterance_gain=gain,transcript_unchanged=True),clean_component=rendered['clean_component'])
                del prepared,rendered
        utterances.append(dict(id=clip['id'],cleaner_ordinal=clip['cleaner_ordinal'],source_id=row['source_id'],dataset=row['dataset'],split='train',
            display_voice_label=clip['display_voice_label'],speaker_identity_verified=False,original_supplied_transcript=row['text'],
            source_audio=source_pin,samples=len(wave),duration_s=len(wave)/16000,common_gain=gain,noise_conditions=clip['noise_conditions']))
        del wave,clean,segments
    assert len(samples)==21 and all(v['decoded_clipped_samples']==0 for v in samples)
    runtime={name:importlib.metadata.version(name) for name in ('numpy','soundfile','scipy')}
    manifest=dict(kind='FastMSS_real_recorded_noise_listening_samples_v1',schema_version=1,created_at_utc=now(),PID=os.getpid(),owner='/root/cpu_data_preparation',
        registry=validated['binding'],source_selection=pins.binding(D/'selected_TRAIN_utterances_v1.json'),mixer_source=mixer_pin,generator_and_decoder_source=pins.binding(__file__),runtime=runtime,
        utterances=utterances,samples=samples,total_audio_examples=21,real_recorded_noise_types=3,clip_RMS_SNR_levels_dB=[20,10,5],
        SNR_is_active_speech_or_VAD_estimate=False,common_linear_gain_per_utterance=True,original_speech_and_recordings_unchanged=True,
        Gaussian_noise_used=False,RIR_used=False,recognizer_or_model_used=False,transcript_correction=False,
        full_WAV_pool_created=False,production_enabled=False,quality_gain_claimed=False)
    write_json(D/'sample_manifest_v1.json',manifest)
    cards=[]
    for number,u in enumerate(utterances,1):
        tracks=[]
        for sample in (v for v in samples if v['utterance_id']==u['id']):
            key=sample['environment'];is_clean=key=='clean';snr=sample['snr_db']
            label='Clean reference' if is_clean else assets[key]['environment']+' · '+str(snr)+' dB'
            measured='' if is_clean else f"Measured {sample['post_FLAC_PCM16_measured_clip_RMS_SNR_dB']:.2f} dB"
            tracks.append(f'<div class="track" data-environment="{key}" data-snr="{snr}"><div class="track-label"><b>{html.escape(label)}</b><small>{measured}</small></div><audio controls preload="none" aria-label="Utterance {number}: {html.escape(label)}"><source src="data:audio/flac;base64,{audio_payloads[sample["id"]]}" type="audio/flac"></audio></div>')
        cards.append(f'<section class="utterance" data-voice="{u["id"]}"><div class="eyebrow">TRAIN UTTERANCE {number} · {u["duration_s"]:.2f}s</div><h2>{html.escape(u["display_voice_label"])}</h2><p class="transcript" lang="tr">{html.escape(u["original_supplied_transcript"])}</p><details><summary>Source &amp; unchanged transcript</summary><p><b>{html.escape(u["dataset"])}</b>, TRAIN ordinal {u["cleaner_ordinal"]}. The supplied source transcript is shown unchanged. Speaker identity has not been independently verified.</p><code>{html.escape(u["source_id"])}</code></details><div class="tracks">'+''.join(tracks)+'</div></section>')
    credits=''.join(f'<li><b>{html.escape(a["environment"])}</b> — <a href="{html.escape(a["source_page_url"])}" target="_blank" rel="noreferrer">{html.escape(a["title"])}</a> by {html.escape(a["creator"])}. <a href="https://creativecommons.org/publicdomain/zero/1.0/" target="_blank" rel="noreferrer">CC0</a>. Public HQ MP3 preview; source SHA256 <code>{a["audio"]["sha256"]}</code>.</li>' for a in assets.values())
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Recorded-noise listening · Turkish TRAIN</title><style>
:root{font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;color:#18263b;background:#edf2f5;--muted:#566779;--line:#dce4eb;--accent:#087e8b}*{box-sizing:border-box}body{margin:0}header{background:#14283b;color:#fff;padding:44px max(5vw,20px) 34px}header .wrap,main{max-width:1160px;margin:auto}h1{font-size:clamp(1.8rem,4vw,3.2rem);line-height:1.12;letter-spacing:-.04em;margin:10px 0 18px}header p{max-width:850px;color:#c8d8e4}.eyebrow{font-size:.73rem;letter-spacing:.13em;font-weight:750;color:#008594}header .eyebrow{color:#73d4cf}.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{background:#294357;border:1px solid #426174;border-radius:100px;padding:6px 12px;font-size:.8rem}main{padding:24px 20px 56px}.controls{position:sticky;top:0;z-index:3;display:flex;flex-wrap:wrap;gap:14px;align-items:end;background:#fff;padding:18px;border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 24px #14283b0d}label{font-size:.77rem;font-weight:700;color:var(--muted)}select,button{display:block;font:inherit;font-size:.88rem;padding:8px 12px;border:1px solid #cbd7e0;border-radius:7px;background:#fff;color:#172f43;min-width:130px}button{cursor:pointer;min-width:auto}button:hover{background:#e6f5f3}.check{display:flex;gap:7px;align-items:center;padding-bottom:10px}.check input{accent-color:var(--accent)}.count{margin-left:auto;color:var(--muted);font-size:.8rem;padding-bottom:10px}.note{background:#dff2f0;border-left:3px solid #0b9298;padding:14px 18px;border-radius:4px;margin:20px 0;font-size:.9rem}.utterance{background:white;border:1px solid var(--line);border-radius:15px;margin:20px 0;padding:26px}h2{font-size:1.2rem;margin:6px 0 14px}.transcript{font-size:1.12rem;line-height:1.75;margin:0 0 14px;padding:17px 20px;background:#f3f6f8;border-radius:8px}details{font-size:.82rem;color:var(--muted);margin-bottom:18px}summary{cursor:pointer}code{font-size:.75rem;overflow-wrap:anywhere}.track{display:grid;grid-template-columns:minmax(170px,1fr) 2fr;gap:12px;align-items:center;padding:13px 0;border-top:1px solid #e4ebf0}.track-label b{font-size:.91rem}.track-label small{display:block;color:var(--muted);font-size:.74rem}audio{width:100%;height:38px}.provenance{font-size:.88rem;background:#fff;border:1px solid var(--line);padding:22px 26px;border-radius:14px}.provenance li{margin:12px 0}.provenance a{color:#087b88}.foot{color:var(--muted);font-size:.8rem;margin-top:18px}[hidden]{display:none!important}@media(max-width:650px){.track{grid-template-columns:1fr;gap:6px}.utterance{padding:18px}.controls{position:static}.count{margin-left:0}.controls label{flex:1}select{width:100%}header{padding-top:30px}}
</style><header><div class="wrap"><div class="eyebrow">FROZEN PILOT REMAINS SEPARATE · LISTENING PREVIEW</div><h1>Turkish speech.<br>Real recorded environments.</h1><p>Compare the same supplied TRAIN transcript with cafe chatter, road traffic and an indoor fan. These are recordings — no Gaussian bed or simulated room response is used here.</p><div class="chips"><span class="chip">3 TRAIN utterances</span><span class="chip">21 playable examples</span><span class="chip">20 / 10 / 5 dB clip-RMS SNR</span><span class="chip">Production disabled</span></div></div></header><main><div class="controls"><label>Utterance<select id="voice"><option value="all">All three</option><option value="tr_1">Utterance 1</option><option value="tr_2">Utterance 2</option><option value="tr_3">Utterance 3</option></select></label><label>Environment<select id="environment"><option value="all">All recordings</option><option value="cafe">Cafe / crowd</option><option value="traffic">Road traffic</option><option value="fan">Indoor fan</option></select></label><label>SNR<select id="snr"><option value="all">All levels</option><option value="20">20 dB</option><option value="10">10 dB</option><option value="5">5 dB</option></select></label><label class="check"><input id="clean" type="checkbox" checked>Keep clean references</label><button id="pause">Pause all</button><button id="reset">Reset</button><span class="count" id="count"></span></div><p class="note"><b>How to compare:</b> lower SNR means more noise. SNR is measured from the entire clip's speech and added-noise RMS, including pauses; it is <b>not</b> a VAD or active-speech estimate. One common linear gain per utterance keeps speech level consistent across its conditions. No limiter, clipping or per-condition normalization is applied. Preview exports use mono16k PCM16 in lossless FLAC; originals are unchanged.</p>'''+''.join(cards)+'''<section class="provenance"><h2>Actual recordings &amp; provenance</h2><p>Each audio-specific primary page declares CC0. The downloaded assets are public lossy HQ MP3 previews of the named recordings, not the authors' original WAV downloads. Bounded excerpts are downmixed and resampled for mixing. All source/audio/code SHA pins, exact crops, gains and decoded-file checks are in the accompanying sample manifest.</p><ul>'''+credits+'''</ul><p>Supplied source transcripts are preserved. No recognizer, transcription correction, training-policy activation or quality-gain claim is involved. This page contains no simulated RIR; any room-simulation pilot is a separate experiment.</p></section><p class="foot">Self-contained: all21 audio exports, styling and controls are embedded. Choose a player to listen; playing a new example pauses the others.</p></main><script>
const byId=id=>document.getElementById(id);function filter(){const voice=byId('voice').value,env=byId('environment').value,snr=byId('snr').value;let shown=0;document.querySelectorAll('.utterance').forEach(card=>{let visible=0;card.querySelectorAll('.track').forEach(track=>{const clean=track.dataset.environment==='clean';const ok=(voice==='all'||card.dataset.voice===voice)&&(clean?byId('clean').checked:(env==='all'||track.dataset.environment===env)&&(snr==='all'||track.dataset.snr===snr));track.hidden=!ok;if(ok)visible++;else track.querySelector('audio').pause()});card.hidden=visible===0;shown+=visible});byId('count').textContent=shown+' / 21 examples'}['voice','environment','snr','clean'].forEach(id=>byId(id).addEventListener('change',filter));byId('pause').onclick=()=>document.querySelectorAll('audio').forEach(a=>a.pause());byId('reset').onclick=()=>{['voice','environment','snr'].forEach(id=>byId(id).value='all');byId('clean').checked=true;filter()};document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));filter();</script></html>'''
    assert document.count('data:audio/flac;base64,')==21
    target.write_text(document,encoding='utf-8');assert target.stat().st_size<20_000_000
    pins.unchanged()
    verification=dict(kind='FastMSS_real_noise_listening_verification_v1',at_utc=now(),PID=os.getpid(),passed=True,
        clips=3,audio_examples=21,all_durations_exact=True,all_encoded_files_independently_redecoded=True,all_finite=True,
        clipped_samples=0,maximum_decoded_peak=max(v['decoded_peak'] for v in samples),
        maximum_post_encoding_SNR_error_dB=max(abs(v['post_FLAC_PCM16_measured_clip_RMS_SNR_dB']-v['snr_db']) for v in samples if v['snr_db'] is not None),
        page=pins.binding(target),sample_manifest=pins.binding(D/'sample_manifest_v1.json'),registry=validated['binding'],
        embedded_audio_payloads=21,external_audio_or_CSS_or_JS_dependencies=0,source_transcripts_unchanged=True,
        browser_playback_verification_by_ROOT_still_required=True,production_enabled=False)
    write_json(D/'verification_v1.json',verification)
    assert not any(n in sys.modules for n in ('torch','transformers','faster_whisper','ctranslate2'))
    print(dict(page=pins.binding(target),sample_manifest=pins.binding(D/'sample_manifest_v1.json'),verification=pins.binding(D/'verification_v1.json'),
        audio_examples=21,actual_owned_tool_exit_still_required=True),flush=True)


if __name__=='__main__':main()
