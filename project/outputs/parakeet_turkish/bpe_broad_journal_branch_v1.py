"""Preserve an uncheckpointed journal tail and resume only a saved prefix.

No optimizer or sampler is advanced here. A new file contains the checkpoint's
authoritative prefix; the original file, including its tail, remains untouched.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from bpe_broad_production_state_v1 import CommitJournal, require

ROOT=Path(r'E:\gpu_training_scratchpad').resolve()

def _path(value):
    path=Path(value).resolve()
    require(path.is_relative_to(ROOT),'Resume journals must remain on E')
    return path

def branch_checkpoint_prefix(saved, destination, *, guard=lambda:None):
    require(set(saved)=={'path','offset','sha256'},'Exact checkpoint prefix binding required')
    source=_path(saved['path']);destination=_path(destination)
    require(source!=destination and source.is_file() and not destination.exists(),'A fresh branch must preserve the old journal')
    require(type(saved['offset']) is int and saved['offset']>=0,'Integer checkpoint offset required')
    before=source.stat();require(saved['offset']<=before.st_size,'Checkpoint journal was truncated')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.building')
    prefix=hashlib.sha256();whole=hashlib.sha256();remaining=saved['offset'];copied=0
    # Exclusive creation preserves even a failed preparation for diagnosis.
    with source.open('rb') as original, temporary.open('xb') as output:
        while block:=original.read(1024**2):
            guard();whole.update(block)
            part=block[:remaining]
            if part:prefix.update(part);output.write(part);copied+=len(part);remaining-=len(part)
        output.flush();os.fsync(output.fileno())
    after=source.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Concurrent journal changes prohibit resume')
    require(remaining==0 and copied==saved['offset'] and prefix.hexdigest()==saved['sha256'],'Checkpoint prefix hash mismatch')
    temporary.replace(destination)
    journal=CommitJournal(destination)
    require(journal.binding()['offset']==saved['offset'] and journal.binding()['sha256']==saved['sha256'],'Copied authoritative prefix differs')
    receipt={'kind':'ASR18_preserved_tail_checkpoint_branch_v1','original_checkpoint_prefix':dict(saved),
        'original_full_journal':{'path':str(source),'bytes':before.st_size,'sha256':whole.hexdigest()},
        'preserved_tail_bytes':before.st_size-saved['offset'],'branch_checkpoint_prefix':journal.binding(),
        'old_journal_modified':False,'tail_replayed_or_sampler_advanced':False}
    return journal,receipt
