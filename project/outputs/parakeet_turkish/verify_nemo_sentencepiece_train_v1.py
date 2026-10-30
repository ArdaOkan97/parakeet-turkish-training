"""Original released SentencePiece vs converted HF BPE on actual TRAIN text.

Extracts only the small tokenizer/model-config tar members into a NEW directory.
No acoustic inference, target/data changes, old artifact writes or CUDA use.
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import tarfile

import sentencepiece as spm
import torch
from tokenizers import Tokenizer
from transformers import AutoProcessor
from extend_vocab import _make_tokenizer
from metrics import normalize_text
from runtime import ROOT,OUTPUT


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['dataset']=='google/fleurs' and row['dataset_split']=='train':yield row


def main():
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(),'CPU-only tokenizer audit')
    archive=ROOT/'work/models/hf_native_cache/models--nvidia--parakeet-tdt-0.6b-v3/snapshots/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/parakeet-tdt-0.6b-v3.nemo'
    extracted=ROOT/'work/research/nemo_reference/tokenizer_original_v1'
    output=OUTPUT/'nemo_sentencepiece_train_parity_v1.json'
    require(not extracted.exists() and not output.exists(),'Preserve earlier tokenizer evidence')
    extracted.mkdir()
    members={}
    with tarfile.open(archive,'r:*') as tar:
        for member in tar:
            basename=Path(member.name).name
            if not (basename.endswith('tokenizer.model') or basename.endswith('tokenizer.vocab') or basename.endswith('vocab.txt') or basename=='model_config.yaml'):continue
            require(member.isfile() and member.size<=1_000_000,'Small regular tokenizer member required')
            data=tar.extractfile(member).read()
            require(len(data)==member.size,'Complete tokenizer member')
            target=extracted/basename
            target.write_bytes(data)
            members[member.name]={'path':str(target),'bytes':member.size,'sha256':sha(target)}
    sp_path=next(Path(value['path']) for key,value in members.items() if key.endswith('tokenizer.model'))
    native=spm.SentencePieceProcessor(model_file=str(sp_path))
    base=(ROOT/'work/base_model_path.txt').read_text().strip()
    tokenizer=AutoProcessor.from_pretrained(base,local_files_only=True).tokenizer
    require(native.get_piece_size()==8192 and len(tokenizer)==8193,'Original vocab sizes')
    require(all(native.id_to_piece(index)==tokenizer.convert_ids_to_tokens(index) for index in range(8192)),
            'Original token IDs/pieces differ')
    extension=ROOT/'work/research/extension_roundtrip/tokenizer.json'
    extended=_make_tokenizer(tokenizer,Tokenizer.from_file(str(extension)))
    prepared=ROOT/'work/data/tr_public_v2_prepared.jsonl'
    replay=ROOT/'work/data/replay_train.jsonl'
    actual=list(rows(prepared));counts=collections.Counter()
    for row in rows(replay):
        if counts[row['language']]<50:actual.append(row);counts[row['language']]+=1
    require(actual and all(row['dataset_split']=='train' for row in actual),'Actual TRAIN texts required')
    summary=collections.defaultdict(collections.Counter);examples=[];piece_segment_differences=[]
    for row in actual:
        text=row['text'];language=row['language'];stats=summary[language]
        sp_ids=native.encode(text,out_type=int)
        hf_ids=tokenizer(text,add_special_tokens=False)['input_ids']
        require(all(0<=value<8192 for value in hf_ids),'Base positive ID bound')
        stats['actual_train_rows']+=1
        stats['original_native_and_hf_encoded_ids_equal']+=(sp_ids==hf_ids)
        sp_text=native.decode(sp_ids)
        hf_text=tokenizer.decode(hf_ids,skip_special_tokens=True,group_tokens=False)
        stats['original_unknown_rows']+=(native.unk_id() in sp_ids)
        stats['known_rows_original_native_and_hf_decode_equal']+=(native.unk_id() not in sp_ids and sp_text==hf_text)
        stats['known_rows']+=(native.unk_id() not in sp_ids)
        ext_ids=extended(text,add_special_tokens=False)['input_ids']
        ungrouped=extended.decode(ext_ids,skip_special_tokens=True,group_tokens=False)
        grouped=extended.decode(ext_ids,skip_special_tokens=True)
        stats['extended_ungrouped_roundtrip_mismatch']+=(normalize_text(text,language)!=normalize_text(ungrouped,language))
        stats['extended_default_ctc_grouping_changes_decoding']+=(grouped!=ungrouped)
        if re.search(r'\d',text):
            stats['digit_rows']+=1
            stats['digit_original_encoded_ids_equal']+=(sp_ids==hf_ids)
            stats['digit_extended_ungrouped_roundtrip_mismatch']+=(normalize_text(text,language)!=normalize_text(ungrouped,language))
            if len(examples)<12 and normalize_text(text,language)!=normalize_text(ungrouped,language):
                examples.append({'source_id':row['source_id'],'language':language,'text':text,
                    'native_original_ids':sp_ids,'hf_original_ids':hf_ids,'native_original_decoded':sp_text,
                    'hf_original_ungrouped_decoded':hf_text,'extended_ungrouped_decoded':ungrouped,
                    'extended_default_ctc_grouped_decoded':grouped})
        # Unknown pieces have different display policies. Compare known original
        # piece runs separately so that unknown glyph rendering cannot obscure
        # a numeric/word-boundary discrepancy in the converted decoder.
        runs=[];run=[]
        for value in sp_ids+[native.unk_id()]:
            if value==native.unk_id():
                if run:runs.append(run)
                run=[]
            else:run.append(value)
        for run in runs:
            if not any(re.search(r'\d',native.id_to_piece(value)) for value in run):continue
            before=native.decode(run);after=tokenizer.decode(run,skip_special_tokens=True,group_tokens=False)
            stats['known_original_digit_piece_runs']+=1
            stats['known_original_digit_piece_runs_decoded_equal']+=(before==after)
            if before!=after and len(piece_segment_differences)<12:
                piece_segment_differences.append({'source_id':row['source_id'],'native':before,'hf_ungrouped':after,'original_token_ids':run})
    require(not torch.cuda.is_initialized(),'Tokenizer audit initialized CUDA')
    report={'schema':1,'kind':'native_sentencepiece_vs_hf_actual_training_tokenizer_audit','passed_execution':True,
        'CUDA_initialized':False,'producer_sha256':sha(__file__),'original_core_piece_ID_identity_verified':True,
        'native_archive':{'path':str(archive),'bytes':archive.stat().st_size,'model_revision':'541d1f99c6b0c3cd0b11a95167540bb8edefd82b',
                          'whole_archive_sha256_measured_here':False,'small_extracted_members':members},
        'inputs_sha256':{str(path):sha(path) for path in (prepared,replay,extension,Path(base)/'tokenizer.json')},
        'per_language':{key:dict(value) for key,value in summary.items()},'digit_roundtrip_examples':examples,
        'known_digit_piece_run_decode_differences':piece_segment_differences,
        'scope':'Actual TRAIN text and released original8192-piece tokenizer only. HFextension adds6chars with preservedoldIDs; originalSentencePiece has no suchextension. No audio/inference/quality claim or target modification.',
        'CTC_grouping_note':'ParakeetTokenizer defaultgroup_tokens=True collapses adjacent identical IDs. TDT label roundtrip must requestFalse; existing frozen inference/benchmark decode calls should be assessed separately before any versioned correction.'}
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'summary':report['per_language'],'output':str(output)},ensure_ascii=False))


if __name__=='__main__':main()
