"""Pinned original Parakeet physical STFT versus valid-mask frame geometry."""
from bpe_broad_production_state_v1 import require

FRONTEND_SOURCE_SHA256='5ec8a9664456edfc0b56c64083bdcdd74dd78364504d35207ab917d1c3b9f575'

def from_samples(samples):
    require(type(samples) is int and samples>=160,'Positive original declared valid-mask geometry required')
    valid=samples//160
    return {'clean16k_samples':samples,'valid_mask_frames':valid,'physical_feature_frames':valid+1,
        'encoder_frames':(valid+7)//8,'actual_finite_normalization_validation_still_required':True,'original_frontend_source_SHA256':FRONTEND_SOURCE_SHA256}

def from_encoder_upper_bound(frames):
    require(type(frames) is int and frames>0,'Positive original encoder frame declaration')
    return {'encoder_frames':frames,'valid_mask_frames_upper_bound':8*frames,
        'physical_feature_frames_upper_bound':8*frames+1,'clean16k_samples_upper_bound':8*frames*160+159,
        'actual_full_waveform_and_mask_match_still_required':True}
