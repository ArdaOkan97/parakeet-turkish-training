"""Pure before-draw declaration/actual comparison; no decode or selection.

The final catalog/labels/declaration index must be independently source-bound
before these row checks are used. Header/interval declaration is not decoded
validation. No row is cropped, repaired, skipped or committed by this module.
"""
from fractions import Fraction
from bpe_broad_production_state_v1 import require

FIELDS=frozenset(('ordinal','source_component','native_sr','native_samples','clean16k_samples','mel_frames','encoder_frames','target_tokens','status'))
COMPONENTS={'alania_yodas3':0,'kiraat':1}

def validate_declared_row(declaration,row,*,full_target_tokens):
    require(type(declaration) is dict and set(declaration)==FIELDS,'Exact final scalar geometry row required')
    require(declaration['status']=='declared' and row['dataset'] in COMPONENTS,'Unknown/invalid declaration remains unresolved, never excluded')
    for key in FIELDS-{'status'}:require(type(declaration[key]) is int,'Scalar declared integer fields cannot be booleans')
    require(type(full_target_tokens) is int and full_target_tokens>0 and type(row['ordinal']) is int
        and declaration['ordinal']==row['ordinal']>0 and declaration['target_tokens']==full_target_tokens,
        'Exact retained ordinal and every original BPE target required')
    component=COMPONENTS[row['dataset']];sr=declaration['native_sr'];n=declaration['native_samples']
    require(declaration['source_component']==component and n>0 and sr==(48000 if component==0 else 24000),
        'Pinned full source rate/count declaration required; unexpected properties require versioned reconciliation')
    clean=(n*16000+sr-1)//sr;mel=clean//160;t=(mel+7)//8
    require(t>0 and (declaration['clean16k_samples'],declaration['mel_frames'],declaration['encoder_frames'])==(clean,mel,t),
            'Canonical full16k/resample/mel/encoder rounding differs')
    if component==0:
        start=Fraction(str(row['start']));end=Fraction(str(row['end']))
        require(start>=0 and end>start and n==round(end*48000)-round(start*48000),
                'Alania original interval declaration differs; no duration quality gate or source trim substitution')
    return dict(declaration)

def verify_actual_row(declaration,row,actual,*,full_target_tokens):
    expected=validate_declared_row(declaration,row,full_target_tokens=full_target_tokens)
    required=('native_sample_rate','native_samples','decoded_sample_rate','decoded_samples','encoder_frames')
    require(all(type(actual.get(key)) is int for key in required),'Available exact integer decoder observations required')
    require((actual['native_sample_rate'],actual['native_samples'],actual['decoded_sample_rate'],actual['decoded_samples'],actual['encoder_frames'])==
        (expected['native_sr'],expected['native_samples'],16000,expected['clean16k_samples'],expected['encoder_frames']),
        'Full decoded native/16k/frame geometry mismatch: stop before draw; preserve source/labels/frontier for versioned reconciliation')
    require(actual['unique_id']==row['unique_id'] and actual['source_id']==row['source_id'],'Actual original source/clip identity differs from declared row')
    return {'kind':'ASR18_before_draw_actual_geometry_matches_declaration_v1','ordinal':row['ordinal'],
        'unique_id':row['unique_id'],'source_id':row['source_id'],'declaration':expected,'actual_native_and_full16k_and_encoder_frame_geometry_matched':True,
        'declared_header_alone_was_not_decode_validation':True,'sampler_or_Adam_frontier_advanced':False,'labels_changed_or_row_excluded':False}
