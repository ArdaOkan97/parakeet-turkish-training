"""PRIVATE codec delta only; real NEW toy auxiliary archive, no native/public arm.

Reuses reviewed private byte-engine setup, with explicit module/guard adapters.
No horizon repetition or old/current files. Frozen checkpoint gzip unchanged.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import lzma
import os
from pathlib import Path
import sys
import time

P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P/'outputs/parakeet_turkish'));sys.path.insert(0,str(Path(__file__).parent))
import bpe_indexed_owned_checkpoint_storage_v3 as m
import validate_BPE_indexed_future_storage_v2 as tiny
from fastmss_row_serial_cpu_v6 import environment,resource_guard,admit
from bpe_broad_production_state_v1 import canonical,digest,require
from bpe_broad_audio_v1 import record
from atomic_io_retry_v1 import write_json

def guard():
    value=resource_guard();require(value['worker_private_memory']['private_bytes']<256_000_000,'Codec fixture256MB')
    return value

def main(output):
    environment();admission=admit(256_000_000,guard=guard);began=time.monotonic();samples=[guard()];checks=[]
    output=Path(output).resolve();require(output.is_relative_to(m.ROOT) and not output.exists(),'Fresh E fixture only');output.mkdir()
    # PRIVATE synthetic protocol setup only. These are not public ROOT proof IO
    # or accepted constructors. All source callbacks still identify the actual
    # reviewed tiny fixture file; only its explicit private backend/guard vary.
    tiny.module=m;tiny.guard=guard
    engine=tiny.initialize_private(output/'PRIVATE_codec_engine')
    def check(name,value):require(value,name);checks.append(name);samples.append(guard())
    def reject(name,operation):
        try:operation()
        except (ValueError,KeyError,TypeError,OSError,lzma.LZMAError):checks.append(name);samples.append(guard());return
        raise ValueError('Expected codec rejection: '+name)
    try:
        intent=engine.reserve_publication(publication_role='completed_window_end')
        plan=m.frozen.new_json(output/'PRIVATE_plan.json',{'kind':'PRIVATE_NO_NATIVE_DRAW'})
        item=engine.reserve_auxiliary_workspace(intent,plan);root=Path(item['workspace'])
        member=m.frozen.new_json(root/'original.json',{'original_raw_text':'same full supplied text','full_targets':[1,300,8710]})
        journal=output/'PRIVATE_journal.jsonl';journal.write_bytes(b'{"PRIVATE":true}\n')
        prefix={'path':str(journal),'offset':journal.stat().st_size,'sha256':record(journal)['sha256']}
        manifest={'kind':'ASR18_future_owned_window_auxiliary_manifest_v1','workspace_marker':item['workspace_marker'],
            'coordinator_plan':plan,'run_identity_sha256':digest(engine.owner['run_identity']),
            'tensor_payloads_audio_features_and_models_included':False,'files':[{'role':'pre_save','file':member}],
            'directories':['.'],'journal_prefix':prefix}
        proof=tiny.owner_proof(engine,output,'PRIVATE_codec_archive_proof',auxiliary_manifest_sha256=digest(manifest),
            all_CPU_auxiliary_producers_actually_closed0=True,native_auxiliary_writer_contexts_returned=True,all_auxiliary_handles_released=True)
        archived=engine.archive_auxiliary(item['id'],manifest,proof,tiny.synthetic_owner_callback)
        path=Path(archived['archive']['path'])
        check('NEW real auxiliary uses XZ; original raw metadata remains live',path.suffix=='.xz' and record(member['path'])==member)
        check('Actual codec header and every original member hash validate',engine._verify_auxiliary_archive(archived)==manifest)
        received=[];resolution=engine.stream_archived_auxiliary_member(item['id'],member,received.append)
        check('Logical member resolution stays byte exact with native mapping API unchanged',resolution['passed'] and hashlib.sha256(b''.join(received)).hexdigest()==member['sha256'])
        check('Frozen checkpoint metadata archive method remains inherited gzip',m._IndexedProtocol._archive is m.frozen._StorageProtocol._archive)
        def consume(path,*,output_limit=34*1024**2,memory_limit=32*1024**2):
            with m._BoundedAuxiliaryXZReader(path,guard=guard,compressed_limit=2*1024**2,
                    output_limit=output_limit,memory_limit=memory_limit) as stream:
                while stream.read(65536):pass
        consume(path);checks.append('One complete CRC64 XZ stream decoded')
        reject('Actual dictionary header cannot exceed configured decoder memory',lambda:consume(path,memory_limit=1024))
        reject('Cumulative decompressed output cap enforced',lambda:consume(path,output_limit=32))
        with m._BoundedAuxiliaryXZReader(path,guard=guard,compressed_limit=2*1024**2,output_limit=34*1024**2) as stream:
            reject('No unbounded or above1MiB decoder read',lambda:stream.read(1024**2+1))
        raw=path.read_bytes()
        truncated=output/'PRIVATE_truncated.xz';truncated.write_bytes(raw[:-5])
        reject('Truncated stream never verifies',lambda:consume(truncated))
        trailing=output/'PRIVATE_trailing.xz';trailing.write_bytes(raw+b'UNACKNOWLEDGED_TAIL')
        reject('Unknown trailing bytes preserved and rejected',lambda:consume(trailing))
        concatenated=output/'PRIVATE_concatenated.xz';concatenated.write_bytes(raw+raw)
        reject('Concatenated extra XZ stream rejected',lambda:consume(concatenated))
        broken=bytearray(raw);broken[-5]^=1;corrupt=output/'PRIVATE_corrupt.xz';corrupt.write_bytes(broken)
        reject('XZ checksum/footer corruption rejected',lambda:consume(corrupt))
        unsupported=output/'PRIVATE_not_XZ.xz';unsupported.write_bytes(b'not an XZ archive')
        reject('Unknown codec header fails closed',lambda:consume(unsupported))
        check('No actual payload/model/native process was created',not list(engine.folder.rglob('*.tensors.bin')))
        check('Original NEW raw member and owned marker remain exact',record(member['path'])==member and record(item['workspace_marker']['path'])==item['workspace_marker'])
        require(time.monotonic()-began<60,'Focused codec fixture60s cap')
        result={'kind':'ASR18_PRIVATE_future_auxiliary_XZ_codec_delta_controls_v1','passed':True,'source':record(__file__),
            'backend_source':record(m.__file__),'private_setup_source':record(tiny.__file__),
            'producer':{'pid':os.getpid(),'ppid':os.getppid()},'checks':checks,'check_count':len(checks),'CPU_admission':admission,
            'observed_private_peak_bytes':max(s['worker_private_memory']['private_bytes'] for s in samples),
            'wall_seconds':time.monotonic()-began,'auxiliary_archive':archived['archive'],'codec':m.AUXILIARY_CODEC,
            'prospective_compress_increment_bytes':256000000,'prospective_read_index_increment_bytes':64000000,
            'public_constructor_native_or_production_proved':False,'horizon_repeated':False,
            'old_current_original_or_diagnostic_files_mutated':False,'production_enabled':False}
        write_json(output/'report.json',result);print(json.dumps({'passed':True,'checks':len(checks),'report':record(output/'report.json')}),flush=True)
    finally:engine.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
