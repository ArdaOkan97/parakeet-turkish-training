"""Primary-source provenance review; no contact, audio download, or admission."""
import hashlib
import json
import pathlib
import urllib.request

import cpu_preparation_guard_v2 as base
from cpu_preparation_guard_v2 import PROJECT,ROOT,now,sha256,write_json

AREA=ROOT/'analysis/CPU_validation_v3'
base.AREA=AREA

def main():
    guard=base.Guard('rights_provenance_review_v3')
    revision='81d802f45a1c1e1f5331260484dacf1fbea81981'
    url=f'https://huggingface.co/datasets/espnet/yodas3/resolve/{revision}/README.md'
    target=AREA/'primary_source_evidence/yodas3_README_pinned.md';target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        guard.wait()
        with urllib.request.urlopen(url,timeout=30) as response:data=response.read(1_048_577)
        if len(data)>1_048_576:raise ValueError('Rights metadata text exceeds1MiB bound')
        target.write_bytes(data)
    text=target.read_text(encoding='utf-8')
    if 'CC-BY-3.0' not in text and 'cc-by-3.0' not in text:raise ValueError('Pinned YODAS license statement missing')
    alania=ROOT/'datasets/alania/metadata';inventory=json.loads((alania/'source_inventory.json').read_text(encoding='utf-8'))
    bindings=[]
    for name in ['README.md','LICENSE.md']:
        path=alania/name;data=path.read_bytes();row=next(r for r in inventory['files'] if r['path']==name)
        git=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if len(data)!=row['size'] or git!=row['oid']:raise ValueError('Pinned Alania legal metadata Git blob differs')
        bindings.append({'path':str(path),'bytes':len(data),'sha256':sha256(path),'Git_blob_oid':git,'revision':inventory['revision']})
    evidence={'pinned_YODAS_README':{'url':url,'path':str(target),'sha256':sha256(target),'revision':revision},'pinned_Alania_legal_metadata':bindings}
    review={'kind':'Alania_YODAS_license_scope_and_underlying_audio_provenance_review_v3','reviewed_at_utc':now(),
        'evidence':evidence,'verdict':'Dataset-level CC-BY audio license claim verified; per-recording underlying commercial rights and attribution chain are not established by our inspected evidence.',
        'license_scopes':[
            {'asset':'Alania captions,descriptors,measurements and non-transcript columns','declared_license':'CC BY4.0','grant_source':'PatientDesk AI pinned LICENSE.md','audio_permission_granted_by_this_layer':False},
            {'asset':'Alania transcript','declared_license':'Follows YODAS/original uploaders CC BY3.0','grant_source':'Pinned Alania LICENSE.md','independent_original_uploader_rights_verification':False},
            {'asset':'YODAS3 distributed underlying audio','declared_license':'CC BY3.0; dataset authors state source recordings were originally released under this license',
             'grant_source':'Pinned dataset README and authors official2026 paper/card','commercial_use_allowed_by_claimed_license_subject_to_terms':True,
             'per_recording_original_creator_URL_license_evidence_bound_in_current_receipts':False}],
        'what_is_established':['Pinned Alania annotation/transcript license split and original-audio disclaimer.',
            'YODAS authors publish a CC BY3.0 dataset-level audio license claim, not only an annotation license.',
            'CC BY3.0 permits commercial reuse/adaptation when applicable license obligations are satisfied.'],
        'what_is_not_established':['For each staged recording, original title,creator,source URL and dated uploader license evidence have not been bound in our admission receipts.',
            'The inspected pinned public metadata schema exposes encrypted recording IDs but no such original attribution/license fields.',
            'An uploader license designation does not by itself verify rights in every included third-party sound,performance or excerpt.',
            'No recording-level commercial-clearance audit or guarantee of all privacy/publicity/moral rights is claimed.'],
        'attribution_provenance_followup':['Preserve Alania attribution and YODAS citation/license/revision.',
            'Before a commercial-clearance claim, bind available original creator,title,source URL,license and modification notices to admitted recordings; document any unresolved rights.',
            'If the source cannot supply a per-recording provenance chain, keep that uncertainty explicit instead of treating annotations CC BY4.0 as an audio permission.'],
        'interpretation_is_inference_from_primary_sources':True,'research_CPU_preparation_may_continue_under_existing_authorization':True,
        'no_license_based_approval_or_training_stop_requested':True,'accepted_TRAIN_hours':0,'GPU_used':False,'paid_service_or_contact':False,
        'primary_sources':[
            {'title':'Alania pinned LICENSE.md','url':'https://huggingface.co/datasets/cloud0day3/alania-speech-captions-tr/blob/8b0fdd976ed9668b64f5ee0696317a8f241ce39c/LICENSE.md'},
            {'title':'YODAS3 author dataset card','url':'https://huggingface.co/datasets/espnet/yodas3'},
            {'title':'YODAS3 author paper abstract','url':'https://www.isca-archive.org/interspeech_2026/chen26d_interspeech.html'},
            {'title':'Creative Commons Attribution3.0 license summary','url':'https://creativecommons.org/licenses/by/3.0/'},
            {'title':'Creative Commons Attribution4.0 license summary','url':'https://creativecommons.org/licenses/by/4.0/'},
            {'title':'YouTube official license and third-party permission guidance','url':'https://support.google.com/youtube/answer/2797468?hl=en'}]}
    write_json(AREA/'rights_provenance_review_v3.json',review)
    guard.status(phase='primary_source_rights_scope_review_complete',actual_CLOSED0=True,commercial_underlying_recording_rights_established=False,
        annotation_license_is_not_audio_license=True,accepted_TRAIN_hours=0,report=str(AREA/'rights_provenance_review_v3.json'))

if __name__=='__main__':main()
