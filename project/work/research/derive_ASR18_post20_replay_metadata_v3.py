from pathlib import Path
source=Path(__file__).with_name('prepare_ASR18_post20_replay_metadata_v2.py')
target=source.with_name('prepare_ASR18_post20_replay_metadata_v3.py')
old=source.read_text();new=old
for key in ('replay_rng','Portuguese_rng'):
    new=new.replace("initial['"+key+"']", "(initial['"+key+"'][0],tuple(initial['"+key+"'][1]),initial['"+key+"'][2])")
assert old!=new and not target.exists()
target.write_text(new,encoding='utf-8',newline='\n')
