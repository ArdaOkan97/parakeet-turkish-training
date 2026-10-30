"""Versioned continuation mirror seeded from the preserved v1 mirror state.

The existing generic exact-resume counter calculation applies to source1773.
This file does not start a mirror when imported. Root owns process replacement.
"""
from __future__ import annotations
from pathlib import Path
import track_continuation_v1 as previous
from cuda_cache_reclaim_v1 import file_sha
from atomic_io_retry_v1 import read_json, write_json

PREVIOUS_SHA256='77fd906215228529753641a29551031aba275700b61b6af9e5ce6a72692bed32'
ROOT=previous.ROOT
resume_context=previous.resume_context
values_for=previous.values_for
cycle_rows=previous.cycle_rows


def follow(args):
    if file_sha(Path(previous.__file__))!=PREVIOUS_SHA256:
        raise RuntimeError('Original generic interval-rate mirror changed')
    destination=args.state_path.resolve()
    if destination!=ROOT/'work/trackio_continuation_mirror_state_v2.json':
        raise ValueError('Use only this versioned mirror state; preserve all old states')
    if not destination.exists():
        source=ROOT/'work/trackio_continuation_mirror_state_v1.json'
        write_json(destination,read_json(source,{}))
    return previous.follow(args)


def parser():
    result=previous.parser()
    result.set_defaults(state_path=ROOT/'work/trackio_continuation_mirror_state_v2.json')
    return result


if __name__=='__main__':
    args=parser().parse_args()
    try:follow(args)
    finally:
        try:previous.trackio.finish()
        except RuntimeError as error:
            if str(error)!='Call trackio.init() before trackio.finish().':raise
