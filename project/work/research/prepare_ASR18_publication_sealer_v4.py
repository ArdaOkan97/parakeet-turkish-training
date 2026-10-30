"""Prepare versioned sealing of the canonical identity refinement."""
from pathlib import Path
P=Path(__file__).resolve().parents[2]
def main():
    old=P/'work/research/seal_ASR18_window_publication_v3.py';new=old.with_name('seal_ASR18_window_publication_v4.py');assert not new.exists()
    text=old.read_text()
    for before,after in (('production_window_CPU_actual_04','production_window_CPU_actual_05'),('production_window_CPU_guard_04','production_window_CPU_guard_05'),
        ('bpe_production_window_coordinator_v3','bpe_production_window_coordinator_v4'),('bpe_streaming_checkpoint_semantics_v1','bpe_streaming_checkpoint_semantics_v2'),
        ('validate_ASR18_production_window_v4','validate_ASR18_production_window_v5'),('report[\'check_count\']==43','report[\'check_count\']==48'),
        ('Actual clean43-control','Actual clean48-control'),("'230cde'","'834300'"),("'checks':43","'checks':48"),
        ('source_review_manifest_v3','source_review_manifest_v4'),('coordinator3_semantics43','coordinator4_semantics48'),
        ('launch_readiness_v22','launch_readiness_v23'),("'actual_controls':43","'actual_controls':48")):
        assert before in text,before;text=text.replace(before,after)
    new.write_text(text);print(str(new))
if __name__=='__main__':main()
