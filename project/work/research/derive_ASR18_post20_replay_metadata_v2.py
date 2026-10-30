from pathlib import Path
source=Path(__file__).with_name('prepare_ASR18_post20_replay_metadata_v1.py')
target=source.with_name('prepare_ASR18_post20_replay_metadata_v2.py')
old=source.read_text();new=old.replace("snapshot={'source_identity':", "snapshot={'source_step':64250,'source_identity':")
assert old!=new and not target.exists()
target.write_text(new,encoding='utf-8',newline='\n')
