"""Add9 simulated-RIR comparisons to immutable v1; preserve common gain fairness."""
import base64,hashlib,html,io,json,math,os,re,sys
from pathlib import Path
from ASR18_continuous_data_guard_v5 import DataIOGuard,ROOT
from cpu_preparation_guard_v2 import now
from finalize_ASR18_declared_training_geometry_v1 import MetadataIO
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from atomic_io_retry_v1 import read_json,write_json
import fastmss_real_noise_v1 as noise
import fastmss_reverb_preview_v1 as reverb
D=ROOT/'analysis/FastMSS_real_recorded_noise_listening_v1'

def main():
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    guard=DataIOGuard('simulated_reverb_recorded_noise_listening_v2',heavy=False,maximum_bytes=200_000_000);guard.wait();pins=MetadataIO(guard)
    target=D/'recorded-noise-reverb-turkish-listening-v2.html';outdir=D/'samples_v2';assert not target.exists() and not outdir.exists();outdir.mkdir()
    old=pins.json(D/'sample_manifest_v1.json');oldqa=pins.json(D/'verification_v1.json');selection=pins.json(D/'selected_TRAIN_utterances_v1.json')
    assert oldqa['passed'] and oldqa['audio_examples']==21
    assert pins.binding(S/'fastmss_real_noise_v1.py')['sha256']=='7a7587027c112533a081e2a825fa670d92b3a6233ec5a5c935069d2baa051cbc'
    reverb_pin=pins.binding(S/'fastmss_reverb_preview_v1.py');assert reverb_pin['sha256']=='285f2722289214081143c070b00c6cd0adae69382a41c91562552abd265618ad'
    rir_assets=reverb.available_RIRs(guard=guard.wait)
    registry=noise.validate_registry(D/'real_noise_registry_v1.json',guard=guard.wait);assets={v['id']:v for v in registry['registry']['assets']}
    oldhtmlpin=pins.binding(D/'recorded-noise-turkish-listening.html');oldhtml=(D/'recorded-noise-turkish-listening.html').read_text(encoding='utf-8')
    style=re.search(r'<style>(.*?)</style>',oldhtml,re.S).group(1)
    waves={};decode={}
    for key,asset in assets.items():
        receipt=next(v['noise_decode'] for v in old['samples'] if v['environment']==key)
        with sf.SoundFile(asset['audio']['path']) as f:
            assert f.samplerate==receipt['native_rate'] and f.channels==receipt['native_channels']
            f.seek(receipt['native_start_frame']);native=f.read(receipt['native_frames'],dtype='float32',always_2d=True);assert native.nbytes<=32*1024**2;guard.pace(native.nbytes)
        mono=np.mean(native,axis=1,dtype=np.float32);del native;sr=receipt['native_rate'];divisor=math.gcd(sr,16000)
        wave=np.ascontiguousarray(resample_poly(mono,16000//divisor,sr//divisor,window=('kaiser',5.0),padtype='constant'),dtype=np.float32);del mono
        assert len(wave)==receipt['resampled_frames'] and hashlib.sha256(memoryview(wave).cast('B')).hexdigest()==receipt['resampled_f32_SHA256']
        waves[key]=wave;decode[key]=receipt
    samples=[];utterances=[];payloads={};new_count=0;reused=0
    def rms(w):return math.sqrt(float(np.dot(w.astype(np.float64),w.astype(np.float64)))/len(w))
    def export(key,wave,metadata,component=None):
        nonlocal new_count
        assert wave.dtype==np.float32 and np.isfinite(wave).all() and float(np.abs(wave).max())<=.950001
        b=io.BytesIO();sf.write(b,wave,16000,format='FLAC',subtype='PCM_16');data=b.getvalue();assert len(data)<2000000
        decoded,sr=sf.read(io.BytesIO(data),dtype='float32');assert sr==16000 and decoded.ndim==1 and len(decoded)==len(wave) and np.isfinite(decoded).all()
        peak=float(np.abs(decoded).max());assert peak<1 and np.count_nonzero(np.abs(decoded)>=1)==0
        if component is not None:
            measured=20*math.log10(rms(component)/rms(decoded-component));assert abs(measured-metadata['snr_db'])<.02;metadata['post_FLAC_PCM16_measured_clip_RMS_SNR_dB']=measured
        path=outdir/(key+'.flac');path.write_bytes(data);guard.pace(len(data));new_count+=1
        item={**metadata,'id':key,'audio':pins.binding(path),'sample_rate':16000,'samples':len(wave),'duration_s':len(wave)/16000,
            'decoded_peak':peak,'decoded_clipped_samples':0,'encoded_format':'FLAC / PCM16, preview-only quantization','duration_exact':True,'decoded_all_finite':True,'v1_audio_byte_exact_reused':False}
        samples.append(item);payloads[key]=base64.b64encode(data).decode('ascii')
    for i,clip in enumerate(selection['clips']):
        u=next(v for v in old['utterances'] if v['id']==clip['id']);row=clip['source_row'];source=pins.binding(row['audio_filepath'],maximum=32_000_000);assert source==u['source_audio']
        with sf.SoundFile(source['path']) as f:
            assert f.samplerate==16000 and f.channels==1 and f.frames==u['samples'];clean=f.read(dtype='float32');guard.pace(clean.nbytes)
        old_samples=[v for v in old['samples'] if v['utterance_id']==u['id']];assert len(old_samples)==7
        reverbs=[reverb.prepare_reverberation(clean,a,guard=guard.wait) for a in rir_assets]
        combo_asset=rir_assets[i%2];combo=reverbs[i%2];noise_key=clip['noise_conditions'][0]
        original_combo=next(v for v in old_samples if v['environment']==noise_key and v['snr_db']==10)
        def segment(key):
            reference=next(v for v in old_samples if v['environment']==key and v['snr_db']==10)
            oldseg=reference['segment'];part,receipt=noise.prepare_noise_segment(waves[key],len(clean),oldseg['seed'],guard=guard.wait,source_offset_samples=oldseg['asset_start_sample']-oldseg['excerpt_start_sample'])
            assert receipt==oldseg;return part,receipt
        seg,segreceipt=segment(noise_key)
        mixed=noise.prepare_mix(combo['audio'],seg,10,guard=guard.wait)
        oldpeaks=[v['mix_receipt']['unscaled_peak'] for v in old_samples if v['snr_db'] is not None]
        peaks=oldpeaks+[v['receipt']['unscaled_peak'] for v in reverbs]+[mixed['receipt']['unscaled_peak']]
        gain=noise.common_gain_for_conditions(clean,peaks,guard=guard.wait,ceiling=.95)
        oldgain=u['common_gain'];assert gain<=oldgain+1e-10
        unchanged=abs(gain-oldgain)<1e-12
        for item in old_samples:
            kind='clean' if item['environment']=='clean' else 'noise';meta={**item,'kind':kind,'noise_environment':None if kind=='clean' else item['environment'],'common_utterance_gain':gain}
            assert pins.binding(item['audio']['path'])==item['audio']
            if unchanged:
                data=Path(item['audio']['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==item['audio']['sha256']
                samples.append({**meta,'v1_audio_byte_exact_reused':True});payloads[item['id']]=base64.b64encode(data).decode('ascii');reused+=1
            elif kind=='clean':export(item['id'],(clean*gain).astype(np.float32),meta)
            else:
                part,receipt=segment(item['environment']);prepared=noise.prepare_mix(clean,part,item['snr_db'],guard=guard.wait);rendered=noise.render_mix(prepared,gain,guard=guard.wait)
                export(item['id'],rendered['audio'],{**meta,'mix_receipt':rendered['receipt']},component=rendered['clean_component']);del prepared,rendered,part
        for position,(asset,prepared) in enumerate(zip(rir_assets,reverbs)):
            rendered=reverb.render_reverberation(prepared,gain,guard=guard.wait)
            export(u['id']+'_rir'+str(position),rendered['audio'],dict(utterance_id=u['id'],kind='reverb',environment='rir_position'+str(position),noise_environment=None,snr_db=None,
                RIR_asset_id=asset['id'],RIR_position=position,reverb_receipt=rendered['receipt'],common_utterance_gain=gain,transcript_unchanged=True))
        rendered=noise.render_mix(mixed,gain,guard=guard.wait)
        export(u['id']+'_rir_noise10',rendered['audio'],dict(utterance_id=u['id'],kind='reverb_noise',environment=noise_key,noise_environment=noise_key,snr_db=10,
            RIR_asset_id=combo_asset['id'],RIR_position=i%2,reverb_receipt=combo['receipt'],segment=segreceipt,noise_decode=decode[noise_key],
            mix_receipt=rendered['receipt'],common_utterance_gain=gain,noise_added_after_reverberation=True,SNR_speech_component='actual reverberant speech after RMS restoration',transcript_unchanged=True),component=rendered['clean_component'])
        utterances.append({**u,'v1_common_gain':oldgain,'common_gain':gain,'v1_audio_reused_unchanged':unchanged,'RIR_and_noise_gain_shared_with_all_old_conditions':True})
        del clean,reverbs,mixed,rendered,seg
    assert len(samples)==len({v['id'] for v in samples})==30 and len([v for v in samples if v['kind']=='reverb'])==6
    manifest=dict(kind='FastMSS_recorded_noise_and_simulated_reverb_listening_samples_v2',schema_version=2,created_at_utc=now(),PID=os.getpid(),owner='/root/cpu_data_preparation',
        v1_manifest=pins.binding(D/'sample_manifest_v1.json'),v1_page=oldhtmlpin,v1_immutable=True,registry=registry['binding'],
        RIR_registry=rir_assets[0]['registry_binding'],RIR_assets=rir_assets,shared_reverb_source=reverb_pin,real_noise_mixer_source=pins.binding(S/'fastmss_real_noise_v1.py'),
        generator_source=pins.binding(__file__),utterances=utterances,samples=samples,total_audio_examples=30,new_RIR_examples=9,
        newly_rendered_encoded_files=new_count,v1_audio_exports_reused_byte_exact=reused,common_gain_across_all_conditions_per_utterance=True,
        RIRs_are_simulated=True,simulated_rooms=1,simulated_source_positions=2,measured_RT60_claimed=False,
        configuration_RT60_range_only_seconds=[.15,.25],alignment='strongest_tap_not_proven_physical_direct_arrival',speech_input_trimmed=False,
        output_length='originalN; post-utterance convolution tail not appended',operation_order='RIR strongesttap/unitL2/convolution/originalN crop/RMS restore, then actual recordednoise at10dB, then one shared gain',
        SNR_basis='wholeclip RMS, not VAD; combination10dB uses actual reverberant speech component',Gaussian_used=False,
        supplied_transcripts_unchanged=True,ASR_or_model_or_GPU_used=False,production_enabled=False,quality_gain_claimed=False)
    write_json(D/'sample_manifest_v2.json',manifest)
    cards=[]
    for number,u in enumerate(utterances,1):
        group=[v for v in samples if v['utterance_id']==u['id']]
        group.sort(key=lambda v:({'clean':0,'reverb':1,'reverb_noise':2,'noise':3}[v['kind']],v.get('RIR_position',0),-int(v['snr_db'] or 0),v['environment']))
        tracks=[]
        for item in group:
            kind=item['kind'];snr=item['snr_db'];label='Clean reference';detail=''
            if kind=='reverb':label='Simulated room 0 · position '+str(item['RIR_position']);detail='Reverb only · no added noise'
            elif kind=='reverb_noise':label='Position '+str(item['RIR_position'])+' + '+assets[item['noise_environment']]['environment']+' · 10 dB';detail='Reverb → recorded noise'
            elif kind=='noise':label=assets[item['environment']]['environment']+' · '+str(snr)+' dB';detail='Recorded noise only'
            if snr is not None:detail+=' · measured '+f"{item['post_FLAC_PCM16_measured_clip_RMS_SNR_dB']:.2f}"+' dB'
            tracks.append(f'<div class="track" data-kind="{kind}" data-snr="{snr}"><div class="track-label"><b>{html.escape(label)}</b><small>{html.escape(detail)}</small></div><audio controls preload="none" aria-label="Utterance {number}: {html.escape(label)}"><source src="data:audio/flac;base64,{payloads[item["id"]]}" type="audio/flac"></audio></div>')
        cards.append(f'<section class="utterance" data-voice="{u["id"]}"><div class="eyebrow">TRAIN UTTERANCE {number} · {u["duration_s"]:.2f}s</div><h2>{html.escape(u["display_voice_label"])}</h2><p class="transcript" lang="tr">{html.escape(u["original_supplied_transcript"])}</p><details><summary>Source &amp; comparison details</summary><p>Supplied TRAIN transcript unchanged. The same clip duration and one shared gain are used for every condition. Simulated responses use strongest-tap alignment, restore clip RMS, and omit the tail beyond the original clip end. Combination noise is added after reverb;10dB is measured against the actual reverberant speech component, including pauses.</p><p>Source: <code>{html.escape(u["source_id"])}</code>. Speaker identity is not independently verified.</p></details><div class="tracks">'+''.join(tracks)+'</div></section>')
    credits=''.join(f'<li><a href="{html.escape(a["source_page_url"])}" target="_blank" rel="noreferrer">{html.escape(a["title"])}</a> by {html.escape(a["creator"])} · <a href="https://creativecommons.org/publicdomain/zero/1.0/">CC0</a>. Public HQ MP3 preview of an actual recording.</li>' for a in assets.values())
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reverb + real noise · Turkish TRAIN</title><style>'''+style+'''</style><header><div class="wrap"><div class="eyebrow">LISTENING PREVIEW V2 · PRODUCTION DISABLED</div><h1>Hear the reverberation.<br>Then add a real environment.</h1><p>The same three Turkish TRAIN clips and supplied transcripts, with clean references, two simulated RIR positions, and recorded environmental noise.</p><div class="chips"><span class="chip">30 playable comparisons</span><span class="chip">1 simulated room · 2 positions</span><span class="chip">Reverb + real noise at10dB</span><span class="chip">Original v1 preserved</span></div></div></header><main><div class="controls"><label>Utterance<select id="voice"><option value="all">All three</option><option value="tr_1">Utterance 1</option><option value="tr_2">Utterance 2</option><option value="tr_3">Utterance 3</option></select></label><label>Comparison<select id="mode"><option value="all">All conditions</option><option value="reverb">Reverb only</option><option value="reverb_noise">Reverb + real noise</option><option value="noise">Real noise only</option></select></label><label>SNR<select id="snr"><option value="all">All levels</option><option value="20">20 dB</option><option value="10">10 dB</option><option value="5">5 dB</option></select></label><label class="check"><input id="clean" type="checkbox" checked>Keep clean references</label><button id="pause">Pause all</button><button id="reset">Reset</button><span class="count" id="count"></span></div><p class="note">Start with <b>Clean reference</b>, then compare <b>position0</b>, <b>position1</b>, and <b>reverb + recorded noise</b>. Lower SNR means more added noise. These two RIRs simulate one room at two source positions; they are distinct from the real cafe, fan and traffic recordings.</p>'''+''.join(cards)+'''<section class="provenance"><h2>Sources &amp; interpretation</h2><p>Recorded noises: audio-specific primary CC0 pages, using the named public lossy HQ MP3 preview assets.</p><ul>'''+credits+'''</ul><details><summary>Simulated RIR provenance and level/tail details</summary><p>The same two previously generated FastMSS RIR assets are reused: room0, sourceposition0 and sourceposition1. Upstream <a href="https://github.com/popcornell/FastMSS/tree/e481ceac821ef0cb31033bf43b45922e377f528d">FastMSS commit e481ceac…</a>, pyroomacoustics0.10.1, seed20261017. Their configuration RT60 range was0.15–0.25s; no measured RT60 is asserted. These are simulated responses, not real room recordings.</p><p>Strongest-tap alignment is not proof of physical direct arrival. The full convolution is kept only through the original sample count and speech RMS is restored; no input speech is removed. Recorded noise follows the RIR. All conditions for one utterance share one final linear gain; no clipping, limiter or per-condition normalization. SNR is whole-clip RMS including pauses, not a speech/VAD estimate. FLAC previews use mono16kPCM16.</p></details><p>No Gaussian bed, ASR correction, training-policy activation or quality-gain claim. The original <a href="recorded-noise-turkish-listening.html">21-example v1</a> and every original asset remain unchanged.</p></section><p class="foot">Self-contained: all30 audio examples, styling and controls are embedded. Playing one example pauses the others.</p></main><script>
const byId=id=>document.getElementById(id);function filter(){const voice=byId('voice').value,mode=byId('mode').value,snr=byId('snr').value;let shown=0;document.querySelectorAll('.utterance').forEach(card=>{let visible=0;card.querySelectorAll('.track').forEach(track=>{const clean=track.dataset.kind==='clean';const ok=(voice==='all'||card.dataset.voice===voice)&&(clean?byId('clean').checked:(mode==='all'||track.dataset.kind===mode)&&(snr==='all'||track.dataset.snr===snr));track.hidden=!ok;if(ok)visible++;else track.querySelector('audio').pause()});card.hidden=visible===0;shown+=visible});byId('count').textContent=shown+' / 30 examples'}['voice','mode','snr','clean'].forEach(id=>byId(id).addEventListener('change',filter));byId('pause').onclick=()=>document.querySelectorAll('audio').forEach(a=>a.pause());byId('reset').onclick=()=>{['voice','mode','snr'].forEach(id=>byId(id).value='all');byId('clean').checked=true;filter()};document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})));filter();</script></html>'''
    assert document.count('data:audio/flac;base64,')==30;target.write_text(document,encoding='utf-8');assert target.stat().st_size<20000000
    pins.unchanged()
    qa=dict(kind='FastMSS_reverb_recorded_noise_listening_verification_v2',at_utc=now(),PID=os.getpid(),passed=True,page=pins.binding(target),sample_manifest=pins.binding(D/'sample_manifest_v2.json'),
        audio_examples=30,new_RIR_comparisons=9,newly_encoded_files_redecoded=new_count,v1_audio_exports_reused_byte_exact=reused,
        v1_page_preserved_byte_exact=True,all_durations_exact=True,clipped_samples=0,maximum_decoded_peak=max(v['decoded_peak'] for v in samples),
        maximum_post_encoding_SNR_error_dB=max(abs(v['post_FLAC_PCM16_measured_clip_RMS_SNR_dB']-v['snr_db']) for v in samples if v['snr_db'] is not None),
        common_gain_per_utterance=True,measured_RT60_claimed=False,source_transcripts_unchanged=True,production_enabled=False,
        browser_verification_by_ROOT_still_required=True)
    write_json(D/'verification_v2.json',qa)
    assert not any(n in sys.modules for n in ('torch','transformers','faster_whisper','ctranslate2'))
    print(dict(page=pins.binding(target),sample_manifest=pins.binding(D/'sample_manifest_v2.json'),verification=pins.binding(D/'verification_v2.json'),newly_encoded=new_count,v1_reused=reused,actual_owned_exit_still_required=True),flush=True)

if __name__=='__main__':main()
