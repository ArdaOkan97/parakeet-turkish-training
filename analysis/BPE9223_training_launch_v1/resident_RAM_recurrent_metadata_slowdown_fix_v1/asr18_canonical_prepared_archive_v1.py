"""INACTIVE representation-only prepared-metadata archive codec.

Canonical bytes replace retained nested Python records after the SAME positive
commit. Original checkpoint location, gzip format and uncompressed JSON bytes
remain unchanged. No numerical package, GC setting, source filter or launcher.
"""
import gzip
from bpe_broad_production_state_v1 import canonical, require


def encode_record(record):
    """The ORIGINAL canonical function, with its original value/error behavior."""
    return canonical(record)


def write_canonical_records(stream, records):
    """Streaming identity: canonical([x,...]) == b'['+canonical(x)+...+b']'.

    Records are internal immutable results of encode_record, not an external
    JSON ingress. No new per-record byte/geometry/sample cap is introduced.
    The stream and every write failure remain owned by the checkpoint caller.
    """
    stream.write(b'[')
    for index, record in enumerate(records):
        require(type(record) is bytes,'Only original canonical immutable bytes in prepared archive')
        if index:stream.write(b',')
        stream.write(record)
    stream.write(b']')


def write_gzip_archive(path, records):
    """The SAME exclusive gzip creation, no extra flush/fsync/publish order."""
    with gzip.open(path,'xb') as stream:
        write_canonical_records(stream,records)
