"""Correct only independent JSON RNG expectation in the new CPU fixture."""
from pathlib import Path
import sys
P=Path(__file__).resolve().parents[2];S=P/'outputs/parakeet_turkish';sys.path.insert(0,str(S))
from bpe_broad_audio_v1 import require
def main():
    p=P/'work/research/validate_ASR18_production_window_v3.py';q=p.with_name('validate_ASR18_production_window_v4.py');require(not q.exists(),'Fresh fixture source')
    text=p.read_text()
    old="""    def semantic(value):return verify_saved_metadata(value,identity=expected_identity,optimizer_updates=21,
        sampler_state=latest['sampler'],journal_prefix=expected_prefix,diagnostic_only=True)"""
    new="""    # Independent JSON pointer serialized exactly two Python RNG tuples as
    # lists. Reconstruct/validate only those fields; all other types stay strict.
    import random
    expected_sampler=copy.deepcopy(latest['sampler'])
    def rng_tuple(value):return tuple(rng_tuple(v) for v in value) if type(value) is list else value
    for key in ('replay_rng','Portuguese_rng'):
        raw=expected_sampler['carried_replay'][key];require(type(raw) is list and len(raw)==3,'Known JSON Python RNG state required')
        converted=rng_tuple(raw);local_rng=random.Random();local_rng.setstate(converted)
        require(local_rng.getstate()==converted,'Independently validated exact Python RNG state')
        expected_sampler['carried_replay'][key]=converted
    check(digest(expected_sampler)==digest(latest['sampler']),'Only known JSON RNG type reconstruction preserves independent sampler content')
    def semantic(value):return verify_saved_metadata(value,identity=expected_identity,optimizer_updates=21,
        sampler_state=expected_sampler,journal_prefix=expected_prefix,diagnostic_only=True)"""
    require(text.count(old)==1,'Exact expected-state fixture delta');q.write_text(text.replace(old,new));print(str(q))
if __name__=='__main__':main()
