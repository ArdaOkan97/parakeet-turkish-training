from pathlib import Path
p=Path(__file__).with_name('validate_ASR18_runtime_geometry_v1.py');q=p.with_name('validate_ASR18_runtime_geometry_v2.py')
old=p.read_text();new=old.replace("row=item['row'];actual=item['reader_evidence'];u=len(row['tr_metadata_labels'])", "training=item['row'];row=training['broad_original'];actual=item['reader_evidence'];u=len(training['tr_metadata_labels'])")
assert old!=new and not q.exists();q.write_text(new,encoding='utf-8',newline='\n')
