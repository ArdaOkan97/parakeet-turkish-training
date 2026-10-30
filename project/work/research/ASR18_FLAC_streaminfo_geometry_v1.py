"""Parse only a FLAC STREAMINFO declaration, never decode or certify audio frames.

Bit layout: RFC9639 section8.2. Unknown total_samples remains unresolved.
Derived lengths match the existing clean16kHz caller's integer rounding contract.
"""
import hashlib


SPECIFICATION = 'https://www.rfc-editor.org/rfc/rfc9639.html#name-streaminfo'


def streaminfo_geometry(prefix):
    prefix = bytes(memoryview(prefix)[:42])
    if len(prefix) != 42 or prefix[:4] != b'fLaC':
        raise ValueError('Unresolved: missing complete native-FLAC STREAMINFO prefix')
    if prefix[4] & 127 != 0 or int.from_bytes(prefix[5:8], 'big') != 34:
        raise ValueError('Unresolved: first metadata block must be34-byte STREAMINFO')
    data = prefix[8:42]
    packed = int.from_bytes(data[10:18], 'big')
    rate = packed >> 44
    channels = ((packed >> 41) & 7) + 1
    bits = ((packed >> 36) & 31) + 1
    samples = packed & ((1 << 36)-1)
    minimum = int.from_bytes(data[:2], 'big')
    maximum = int.from_bytes(data[2:4], 'big')
    if not 0 < rate <= 1048575 or not 4 <= bits <= 32 or not 16 <= minimum <= maximum <= 65535:
        raise ValueError('Unresolved: invalid STREAMINFO scalar properties')
    output = (samples*16000 + rate-1)//rate if samples else None
    mel = output//160 if output is not None else None
    return {'sample_rate': rate, 'channels': channels, 'bits_per_sample': bits,
            'total_native_samples': samples, 'known_total_samples': samples > 0,
            'clean16k_samples_from_header': output,
            'mel_frames_from_header': mel,
            'encoder_frames_from_header': (mel+7)//8 if mel is not None else None,
            'minimum_blocksize': minimum, 'maximum_blocksize': maximum,
            'unencoded_MD5': data[18:34].hex(),
            'STREAMINFO_prefix_SHA256': hashlib.sha256(prefix).hexdigest(),
            'geometry_basis': 'FLAC_STREAMINFO_declaration_not_decoded_frame_validation',
            'native24k_mono_as_dataset_card': rate == 24000 and channels == 1,
            'runtime_full_decode_length_check_still_required': True}
