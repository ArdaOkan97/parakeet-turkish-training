"""Separately bounded full on-disk compact frontier decode; no audio."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(r'E:\gpu_training_scratchpad')
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scratch/CPU_validation_v2_env/Lib/site-packages'))
import cpu_preparation_guard_v2 as base
import paced_CPU_audio_v3 as paced
from finalize_reference_boundary_CPU_v21 import MetadataGuard
from run_large_v3_sanity_cpu_v6 import hard_process_memory_job
import reference_frontier_codec_v21 as codec

class VerifyGuard(MetadataGuard):
    def __init__(self):
        base.Guard.__init__(self,'compact_frontier_full_read_verification_v21')
        self.proof_verified=False;self.last_poll=0.;self.last_snapshot=self.last_reason=None
        self.paused_seconds=0.;self.file_bytes_read=0;self.io_rate=1048576

def main():
    area=ROOT/'analysis/CPU_validation_v21';base.AREA=paced.AREA=area
    guard=VerifyGuard();guard.wait();job=hard_process_memory_job()
    path=area/'actual_next10k_reference_CLOSED0_v21.json'
    closed=json.loads(guard.read_file(path).decode('utf-8'))
    assert closed['exit_code']==0 and closed['actual_reference_clips']==130000
    assert hashlib.sha256(Path(codec.__file__).read_bytes()).hexdigest()==closed['frontier_decoder_source_SHA256']
    count=0;canonical=hashlib.sha256()
    for row in codec.iter_frontier_rows(closed['committed_reference_frontier_path'],pace=guard.pace):
        canonical.update((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8'))
        count+=1
    assert count==130000 and canonical.hexdigest()==closed['frontier_canonical_JSONL_equivalent_SHA256']
    result={'kind':'actual_full_on_disk_compact_frontier_decode_verification_v21','completed_at_utc':base.now(),'PID':os.getpid(),
      'verified_rows':count,'all8metadata_fields_preserved':True,'canonical_JSONL_equivalent_SHA256':canonical.hexdigest(),
      'actual_CLOSED0_receipt_path':str(path),'actual_CLOSED0_receipt_SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'frontier_path':closed['committed_reference_frontier_path'],'frontier_bytes':closed['compact_frontier_encoded_bytes'],
      'compact_decoder_source':str(Path(codec.__file__)),'compact_decoder_source_SHA256':closed['frontier_decoder_source_SHA256'],
      'metadata_paced_bytes_read':guard.file_bytes_read,'maximum_metadata_read_bytes':30_000_000,'read_rate_bytes_s':1048576,
      'CPU_hard_memory_limit_bytes':2_000_000_000,'system_resource_and_GPU_state_guards_unchanged':True,
      'previous_seals_untouched':True,'audio_reads':0,'GPU_used':False,'accepted_TRAIN_hours':0}
    base.write_json(area/'compact_frontier_full_decode_verification_v21.json',result)
    guard.status(phase='completed_full_on_disk_compact_frontier_decode_verification',verified_rows=count)
    print(json.dumps(result))
if __name__=='__main__':main()
