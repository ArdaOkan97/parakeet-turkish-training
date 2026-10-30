"""Inactive first-window metadata planner from final DATA and original sources.

No model, waveform, frontend, feature cache or native commit is created. Final
DATA/declared geometry actual0 evidence is required before opening source rows.
Window2 must instead start from the actual coherent live/restored schedule.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib,json,struct
from pathlib import Path
from bpe_broad_audio_v1 import record,require
from bpe_broad_production_state_v1 import CarriedReplay,digest
from bpe_training_schedule_v1 import TrainingSchedule,WEAK
from bpe_replay_bucket_no_runtime_v1 import BucketCycler
from bpe_original_schedule_snapshot_v1 import load_original_snapshot
from bpe_replay_cache_labels_v1 import full_replay_labels
from bpe_declared_geometry_index_v1 import DeclaredGeometryIndex,open_final,signature
from bpe_windows_final_metadata_lock_v1 import FinalMetadataReadLocks
from asr18_physical_feature_geometry_v2 import from_samples,from_encoder_upper_bound
from bpe_production_window_coordinator_v4 import make_plan
from fastmss_phase_aware_plan_bridge_v3 import validate_DATA,actual0,read_bound


class FinalBroadRows:
    """Bounded raw-row/label resolver, after independent final DATA validation."""
    def __init__(self,validated,*,guard=lambda:None):
        require(validated['kind']=='ASR18_validated_final_DATA_population_for_bounded_preparation_v1'
            and validated['passed'] is True and validated['complete'] is True,'Separately validated complete DATA required')
        self.guard=guard;self.poisoned=False;self.cache=OrderedDict();self.connections=[]
        bindings=validated['bindings'];self.population=validated['retained_rows'];self.signatures=[]
        self.metadata_locks=FinalMetadataReadLocks([bindings[k]['path'] for k in ('final_catalog','final_labels','declared_geometry_index')])
        try:
            for key in ('final_catalog','final_labels'):
                pin=bindings[key];guard();require(record(pin['path'])==pin,'Exact final whole catalog/label hash required')
                self.signatures.append(signature(pin['path']));self.connections.append(open_final(pin['path']))
            self.geometry=DeclaredGeometryIndex(bindings['declared_geometry_index'],guard=guard)
        except BaseException:
            for connection in self.connections:connection.close()
            self.metadata_locks.close()
            raise

    def row(self,ordinal):
        require(not self.poisoned and type(ordinal) is int and 1<=ordinal<=self.population,'Unpoisoned exact final source ordinal required')
        try:
            self.guard();require(all(signature(pin['path'])==pin for pin in self.signatures),'Final DB changed before provisional draw')
            if ordinal in self.cache:
                value=self.cache.pop(ordinal);self.cache[ordinal]=value;return deepcopy(value)
            raw=self.connections[0].execute('SELECT * FROM clips WHERE ordinal=?',(ordinal,)).fetchone()
            label=self.connections[1].execute('SELECT target_ids_u32,target_tokens,text_sha256 FROM labels WHERE ordinal=?',(ordinal,)).fetchone()
            require(raw is not None and label is not None,'Missing retained ordinal stays unresolved')
            raw=dict(raw);blob,u,text_sha=label
            require(type(raw['text']) is str and hashlib.sha256(raw['text'].encode('utf-8')).hexdigest()==text_sha
                and type(blob) is bytes and type(u) is int and u>0 and len(blob)==4*u,'Full raw text/target BLOB must agree')
            ids=[value[0] for value in struct.iter_unpack('<I',blob)]
            require(len(ids)==u and all(v<8711 and v!=8192 for v in ids),'Every supplied full8711 target required')
            declared=self.geometry.declaration(raw,full_target_tokens=u)
            value={**raw,'broad_original':deepcopy(raw),'recording_source_id':raw['source_id'],'source_id':raw['unique_id'],
                'language':'tr','augmentation_role':'student_tr','encoder_frames':declared['encoder_frames'],'target_tokens':u,
                'tr_metadata_labels':ids,'training_input_type':'TR_frame_target_metadata_v1','declared_geometry':declared,
                'runtime_full_audio_geometry_and_heldout_check_required':True,
                'declared_frontend_geometry':from_samples(declared['clean16k_samples'])}
            # Original text/raw/spoken strings and original_flags_json remain
            # byte-semantic metadata. No flag is consulted for admission/order.
            self.cache[ordinal]=deepcopy(value)
            while len(self.cache)>32:self.cache.popitem(last=False)
            return value
        except BaseException:self.poisoned=True;raise

    def close(self):
        for connection in self.connections:connection.close()
        if hasattr(self,'geometry'):self.geometry.close()
        self.metadata_locks.close()


def feature_upper_bound(phase,rows):
    require(phase==WEAK and 3<=len(rows)<=16,'This first-prefix planner supports the weak phase only')
    total=0
    for row in rows:
        t=row['encoder_frames'];require(type(t) is int and t>0,'Declared full encoder geometry required')
        if row['language']=='tr':
            shape=from_samples(row['declared_geometry']['clean16k_samples'])
            require(shape['valid_mask_frames']==row['declared_geometry']['mel_frames'] and shape['encoder_frames']==t,'Exact declared valid frames differ')
            frames=shape['physical_feature_frames']
        else:frames=from_encoder_upper_bound(t)['physical_feature_frames_upper_bound']
        require(type(frames) is int and frames>0,'Conservative physical frontend frame bound required')
        total+=frames*(128*4+1)+2*4096 # two full NPY arrays, bounded headers
    return total


def prepare_first_plan(*,DATA_evidence,snapshot_binding,snapshot_actual0,snapshot_report,
                       expected_original_sources,replay_pool_binding,cleaner_metadata_binding,
                       maximum_updates=20,guard=lambda:None):
    """Return a provisional original-source prefix; never extend a fixture cursor."""
    require(type(maximum_updates) is int and 1<=maximum_updates<=20,'Finite first window maximum20')
    validated=validate_DATA(DATA_evidence,guard) # genuine final positive path, no partial fallback
    actual0(snapshot_actual0,snapshot_report,guard,owner='/root/memory_diagnosis')
    report=read_bound(snapshot_report,guard)
    require(report['passed'] is True and report['snapshot']==snapshot_binding
        and report['all_original_RNG_queues_counts_exact'] is True and report['actual_optimizer_updates']==0
        and report['draws_consumed']==0,'Actual original scalar-source extraction proof required')
    snapshot=load_original_snapshot(snapshot_binding,expected_sources=expected_original_sources,guard=guard)
    require(replay_pool_binding['sha256']=='00b839c717762aba74326e9867969d85b7ffaaabd7ead7ddbe732fe6be8c6cf4'
        and record(replay_pool_binding['path'])==replay_pool_binding,'Whole original replay pool identity required')
    pool=json.loads(Path(replay_pool_binding['path']).read_text(encoding='utf-8'))['replay']
    require(type(pool) is list and len(pool)==1914,'All original1914 replay rows retained')
    require(cleaner_metadata_binding['sha256']=='4a1c2bf698e038f2ea7c886ccf14281e197a01ac77836c8134c9b5b4b010b99a'
        and record(cleaner_metadata_binding['path'])==cleaner_metadata_binding,'Exact independent complete cleaner metadata proof required')
    cleaner=json.loads(Path(cleaner_metadata_binding['path']).read_text(encoding='utf-8'))
    require(cleaner['passed'] is True and cleaner['identity']['rows']==352768
        and cleaner['identity']['full_target_tokens']==11972886,'Fixed complete cleaner denominator required')
    identity={'kind':'ASR18_final_unfiltered_broad_DATA_identity_v1','population':DATA_evidence['population'],
        **{k:validated['bindings'][k] for k in ('final_catalog','final_labels','declared_geometry_index','completed_source_inventory')}}
    def new_schedule():
        return TrainingSchedule(broad_population=validated['retained_rows'],broad_identity=identity,
            broad_mean_U=validated['population_mean_U'],cleaner_population=352768,cleaner_identity=cleaner['identity'],
            cleaner_mean_U=11972886/352768,source_turkish_rng=snapshot['turkish_rng'],
            replay=CarriedReplay(pool,snapshot,BucketCycler))
    schedule=new_schedule();require(schedule.committed_updates()==0 and schedule.phase==WEAK,'Original source fork starts with zero new updates')
    before=deepcopy(schedule.state_dict());resolver=FinalBroadRows(validated,guard=guard);replay_evidence={}
    def resolve_identity(phase,ordinal):
        require(phase==WEAK,'Cleaner requires its separately verified full-row resolver')
        row=resolver.row(ordinal);return {'ordinal':ordinal,'source_id':row['source_id'],'language':'tr'}
    def resolve_rows(phase,draws,replay):
        require(phase==WEAK,'No implicit production cleaner transition from a bounded prefix')
        rows=[resolver.row(draw['ordinal']) for draw in draws]
        for original in replay:
            guard();cache=record(original['cache_path']);ids,proof=full_replay_labels(original,cache,guard=guard)
            key=original['source_id'];require(key not in replay_evidence or replay_evidence[key]==proof,'Replay source/cache binding changed during plan')
            replay_evidence[key]=proof;rows.append({**original,'augmentation_role':'replay','clean_target_ids':ids,'full_replay_label_binding':proof})
        return rows
    try:
        plan=make_plan(schedule,clone_schedule=new_schedule,resolve_identity=resolve_identity,resolve_rows=resolve_rows,
            feature_bytes_upper_bound=feature_upper_bound,finalized_population=DATA_evidence['population'],
            verify_finalized_population=lambda pin:pin==DATA_evidence['population'],maximum_updates=maximum_updates)
        require(schedule.state_dict()==before and schedule.committed_updates()==0,'Provisional original plan must not consume authoritative schedule state')
    finally:resolver.close()
    return {'plan':plan,'initial_sampler':before,'scientific_source_state':snapshot_binding,'original_source_records':expected_original_sources,
        'replay_metadata_label_proofs':replay_evidence,'DATA_evidence':DATA_evidence,'validated_DATA':validated,
        'source':record(__file__),'maximum_windows':2,'maximum_native_updates':40,
        'actual_optimizer_updates':0,'actual_authoritative_draws_consumed':0,'actual_waveforms_or_features_prepared':False,
        'runtime_actual_geometry_and_heldout_checks_still_required':True,'next_window_requires_actual_coherent_checkpoint_and_retirement':True,
        'no_diagnostic20_or21_checkpoint_used_as_source':True,'production_enabled':False}
