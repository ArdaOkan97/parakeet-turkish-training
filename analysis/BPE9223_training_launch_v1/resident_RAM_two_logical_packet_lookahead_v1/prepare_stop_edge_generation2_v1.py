from pathlib import Path
F=Path('E:/gpu_training_scratchpad/analysis/BPE9223_training_launch_v1/resident_RAM_two_logical_packet_lookahead_v1')
p=F/'freeze_checkpoint_stop_edge_successors_v1.py';text=p.read_text(encoding='utf-8-sig')
old="""        native_poststep=('in native_update'in trace and 'norm=float(optimizer.step())'in trace
            and 'run_ASR18_native_throughput_component_v15.py'in trace)
        require(journal_guard or native_poststep,
            'One unpublished counter requires exact original pre-append or post-step cleanup cooperative guard trace')
        site='journal_preappend_original_guard'if journal_guard else'native_poststep_original_cleanup_guard'"""
new="""        require(journal_guard,
            'One unpublished counter requires exact original commit-to-journal pre-append cooperative guard trace')
        site='journal_preappend_original_guard'"""
assert text.count(old)==1;text=text.replace(old,new)
text=text.replace('asr18_approved_lookahead_checkpoint_transition_v2','asr18_approved_lookahead_checkpoint_transition_v3')
old=""" ast.parse(text);assert not Path(new).exists();Path(new).write_bytes(text.replace('\\n','\\r\\n').encode())
 inverse=text
 for a,b,n in reversed(edits):assert inverse.count(b)==n,(b[:100],inverse.count(b),n);inverse=inverse.replace(b,a)
 assert inverse.replace('\\n','\\r\\n').encode()==raw.lstrip(b'\\xef\\xbb\\xbf')
 # BOM in Set-Content sources is declared and restored independently.
 restored=inverse.replace('\\n','\\r\\n').encode()
 if raw.startswith(b'\\xef\\xbb\\xbf'):restored=b'\\xef\\xbb\\xbf'+restored
 assert restored==raw"""
new=""" ast.parse(text);assert not Path(new).exists()
 inverse=text
 for a,b,n in reversed(edits):assert inverse.count(b)==n,(b[:100],inverse.count(b),n);inverse=inverse.replace(b,a)
 original=raw.decode('utf-8-sig');original_lines=original.splitlines(True)
 endings=['CRLF'if x.endswith('\\r\\n')else'LF'if x.endswith('\\n')else'NONE'for x in original_lines]
 bodies=inverse.splitlines(True);assert len(bodies)==len(endings)
 restored=''.join(x.removesuffix('\\n')+('\\r\\n'if ending=='CRLF'else'\\n'if ending=='LF'else'')for x,ending in zip(bodies,endings)).encode()
 if raw.startswith(b'\\xef\\xbb\\xbf'):restored=b'\\xef\\xbb\\xbf'+restored
 assert restored==raw
 Path(new).write_bytes(text.replace('\\n','\\r\\n').encode())"""
assert text.count(old)==1;text=text.replace(old,new)
text=text.replace("'old_and_new_newlines':'CRLF','edits_on_LF_normalized_text'", "'old_line_endings_sequence':endings,'new_line_endings':'CRLF','edits_on_LF_normalized_text'")
text=text.replace("'source_backed_cooperative_stop_site':site,", "'source_backed_cooperative_stop_site':site,\n        'unjournaled_objective_gradient_full_native_receipt':'UNKNOWN_UNPERSISTED'if gap else'NOT_APPLICABLE',")
(F/'freeze_checkpoint_stop_edge_successors_v2.py').write_text(text,encoding='utf-8')
