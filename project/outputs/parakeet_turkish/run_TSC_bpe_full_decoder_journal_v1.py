"""Thin E:-only entry: decoder worker journals from its first admitted start.

There is no runtime monkeypatch or alternate scientific identity in this family.
"""
from run_TSC_bpe_full_decoder_v1 import main, parser

if __name__ == "__main__":
    main(parser().parse_args())
