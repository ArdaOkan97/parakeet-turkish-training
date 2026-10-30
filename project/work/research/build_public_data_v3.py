"""Create only NEW future-data versions; refuse every existing target."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'outputs/parakeet_turkish'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def create(name,text):
    path=D/name
    if path.exists():raise FileExistsError(path)
    ast.parse(text)
    path.write_text(text,encoding='utf-8',newline='\n')
    return sha(path)
def replace_once(text,left,right):
    if text.count(left)!=1:raise ValueError('Unique span required: '+left[:90])
    return text.replace(left,right)

proof_sha=sha(D/'tr_public_v3_preparation.json')
labels_sha=sha(D/'tr_public_v3_tokenizer_acceptance.json')
audit_sha=sha(D/'tr_public_v2_input_verification.json')
text=(D/'public_turkish_data_policy_v2.py').read_text(encoding='utf-8')
text=text.replace('Turkish v2 pool','Turkish v3 pool').replace('source_isolation_v2','source_isolation_v3')
text=text.replace("PREPARATION_SHA = 'ea6a34c8fbaca0912293772cf83c1f20b8c307d4c97aa44378ded60118477720'",f'PREPARATION_SHA = {proof_sha!r}')
text=text.replace("LABEL_PROOF_SHA = '88d2faa208e375ca9c909dbf89e3abecfa6a407cb278649af3b98ec75b6bdad5'",f'LABEL_PROOF_SHA = {labels_sha!r}\nV2_FULL_AUDIO_AUDIT_SHA = {audit_sha!r}')
text=text.replace("'version': 2", "'version': 3, 'tokenizer_decode_group_tokens': False")
text=text.replace('no implicit subset or supplemental source','no implicit subset or supplemental source')
text=text.replace('full vetted v2 pool','full vetted v3 pool').replace('completed v2 preparation','completed v3 preparation').replace('v2 accepted manifest','v3 accepted manifest')
text=text.replace("tr_public_v2_tokenizer_acceptance.json", "tr_public_v3_tokenizer_acceptance.json")
text=text.replace("not labels['existing_actual_trainer_source_set_exact']", "not labels['existing_actual_trainer_source_set_preserved'] or labels.get('group_tokens') is not False")
text=replace_once(text,'*, train_limit=None, supplement=None):','*, train_limit=None, supplement=None, verify_actual_audio=True):')
needle="    heldout = []\n"
insert="""    if proof.get('tokenizer_decode_group_tokens') is not False or proof.get('vocabulary_changed') is not False:
        raise ValueError('Explicit unchanged-vocabulary ungrouped TDT policy required')
    previous_audit_path = Path(proof['reused_full_v2_audio_audit_path'])
    if sha256(previous_audit_path) != V2_FULL_AUDIO_AUDIT_SHA or proof['reused_full_v2_audio_audit_sha256'] != V2_FULL_AUDIO_AUDIT_SHA:
        raise ValueError('Pinned completed v2 full actual-PCM audit differs')
    previous = json.loads(previous_audit_path.read_text(encoding='utf-8'))
    if not previous.get('passed') or previous['audit']['identity']['accepted_clips'] != 71684:
        raise ValueError('Complete previous actual-PCM audit required')
    previous_manifest = Path(proof['prior_v2_manifest_path'])
    if sha256(previous_manifest) != proof['prior_v2_manifest_sha256'] or previous['audit']['identity']['accepted_manifest_sha256'] != proof['prior_v2_manifest_sha256']:
        raise ValueError('Reused actual-audio manifest differs')
    old_rows = read_rows(previous_manifest)
    old_by_id = {row['source_id']: row for row in old_rows}
    tr_by_id = {row['source_id']: row for row in tr}
    if any(tr_by_id.get(source_id) != row for source_id, row in old_by_id.items()):
        raise ValueError('Previously verified audio rows must remain exact')
    recovery = proof['newly_recovered_actual_audio_evidence']
    recovery_by_id = {row['source_id']: row for row in recovery}
    if len(recovery_by_id) != len(recovery) or set(recovery_by_id) != set(tr_by_id)-set(old_by_id):
        raise ValueError('Newly recovered actual-audio evidence must cover the exact added set')
    for source_id, audio in recovery_by_id.items():
        if audio['wav_sha256'] != tr_by_id[source_id]['wav_sha256'] or audio['canonical_pcm_sha256'] != by_id[source_id]['canonical_pcm_sha256']:
            raise ValueError('Recovered WAV/canonical proof differs')
    if len(old_by_id) != proof['reused_verified_audio_clips'] or len(recovery) != proof['newly_recovered_clips']:
        raise ValueError('Actual-audio proof partition counts differ')
"""
text=replace_once(text,needle,insert+needle)
text=replace_once(text,"    heldout_pcm = {audio_evidence(row, training=False) for row in {r['wav_sha256']:r for r in heldout}.values()}","    heldout_pcm = ({audio_evidence(row, training=False) for row in {r['wav_sha256']:r for r in heldout}.values()} if verify_actual_audio else set())")
text=replace_once(text,"        canonical = audio_evidence(row, label['canonical_pcm_sha256'])","        canonical = audio_evidence(row, label['canonical_pcm_sha256']) if verify_actual_audio else label['canonical_pcm_sha256']\n        if not .5 <= row['duration'] <= 30. or not Path(row['audio_filepath']).is_file():\n            raise ValueError('Vetted duration or audio file unavailable')")
text=text.replace('public v2 PCM/input verification','public v3 PCM/input verification').replace('Full v2 accepted counts','Full v3 accepted counts').replace('public_turkish_v2_trainer_input_audit','public_turkish_v3_trainer_input_audit').replace('vetted public v2 pool','vetted public v3 pool')
text=replace_once(text,"            'elapsed_seconds': time.perf_counter() - started,","            'elapsed_seconds': time.perf_counter() - started,\n            'actual_WAVs_reread_in_this_audit': verify_actual_audio,\n            'audio_verification_mode': 'full_actual_PCM_now' if verify_actual_audio else 'hash_bound_completed_v2_full_audio_plus_v3_recovered_actual_audio',\n            'audio_mutation_since_reused_audit_not_claimed_checked': not verify_actual_audio,\n            'reused_actual_audio_audit_sha256': V2_FULL_AUDIO_AUDIT_SHA,")
policy_sha=create('public_turkish_data_policy_v3.py',text)

text=(D/'train_lora_batched_data_v2.py').read_text(encoding='utf-8')
text=text.replace('public_turkish_data_policy_v2','public_turkish_data_policy_v3').replace('source_isolation_v2','source_isolation_v3')
text=text.replace('data_v2_proof','data_v3_proof').replace('--data-v2-proof','--data-v3-proof')
text=text.replace('train_lora_batched_data_v2.py','train_lora_batched_data_v3.py').replace('public_data_v2_acceptance.json','public_data_v3_acceptance.json')
text=text.replace('tr_public_v2_train.jsonl','tr_public_v3_train.jsonl').replace('tr_public_v2_tokenizer_acceptance.json','tr_public_v3_tokenizer_acceptance.json').replace('tr_public_v2_preparation.json','tr_public_v3_preparation.json')
text=replace_once(text,'processor.tokenizer.decode(ids, skip_special_tokens=True)','processor.tokenizer.decode(ids, skip_special_tokens=True, group_tokens=False)')
tree=ast.parse(text)
sync=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.cuda.synchronize']
if len(sync)!=1 or sync[0].lineno!=539:raise ValueError('Reviewed update hook line must remain539')
trainer_sha=create('train_lora_batched_data_v3.py',text)

text=(D/'launch_with_cache_reclaim_v2.py').read_text(encoding='utf-8').replace('data-v2','data-v3').replace('train_lora_batched_data_v2.py','train_lora_batched_data_v3.py')
text=text.replace('9f158449e181768489c64dfbe9060c81c4166ec4a8999340616124bf4e8f6f5a',trainer_sha).replace('cache_reclamation_training_launcher_v2','cache_reclamation_training_launcher_v3')
cache_sha=create('launch_with_cache_reclaim_v3.py',text)

text=(D/'launch_with_memory_guard_v4.py').read_text(encoding='utf-8').replace('data-v2','data-v3')
replacements={
 'train_lora_batched_data_v2.py':'train_lora_batched_data_v3.py',
 'public_turkish_data_policy_v2.py':'public_turkish_data_policy_v3.py',
 'launch_with_cache_reclaim_v2':'launch_with_cache_reclaim_v3',
 'resource_watchdog_v1':'resource_watchdog_v2',
 '9f158449e181768489c64dfbe9060c81c4166ec4a8999340616124bf4e8f6f5a':trainer_sha,
 'bd0525a260edf6d4de16cc8b74c5e8ed180d00466a47b2adb063d776a907cd1c':policy_sha,
 'e4785bbcac64a50a82d36a3a3ba05c1999bdb096f0483ef57097cf37f44b6415':cache_sha,
 'cc4ca1f11fdef19ac38d2aa878334451a100fd79f9a2e25a5afb4ceef89f0d7c':sha(D/'resource_watchdog_v2.py'),
 'memory_guard_training_launcher_v4':'memory_guard_training_launcher_v5',
}
for left,right in replacements.items():text=text.replace(left,right)
text=replace_once(text,"RESOURCE_HELPER_SHA256 = "+repr(sha(D/'resource_watchdog_v2.py')),
 "RESOURCE_HELPER_SHA256 = "+repr(sha(D/'resource_watchdog_v2.py'))+"\nRESOURCE_TELEMETRY_SHA256 = "+repr(sha(D/'paired_windows_commit_telemetry_v1.py')))
text=replace_once(text,"            ('resource_watchdog_v2.py', RESOURCE_HELPER_SHA256),","            ('resource_watchdog_v2.py', RESOURCE_HELPER_SHA256),\n            ('paired_windows_commit_telemetry_v1.py', RESOURCE_TELEMETRY_SHA256),")
text=replace_once(text,"               'cache_helper_sha256': CACHE_HELPER_SHA256, 'resource_helper_sha256': RESOURCE_HELPER_SHA256,","               'cache_helper_sha256': CACHE_HELPER_SHA256, 'resource_helper_sha256': RESOURCE_HELPER_SHA256,\n               'resource_telemetry_sha256': RESOURCE_TELEMETRY_SHA256,")
wrapper_sha=create('launch_with_memory_guard_v5.py',text)
print(json.dumps(dict(policy_sha256=policy_sha,trainer_sha256=trainer_sha,cache_launcher_sha256=cache_sha,memory_launcher_sha256=wrapper_sha)))
