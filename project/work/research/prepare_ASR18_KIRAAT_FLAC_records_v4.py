"""Unchanged append-only/full-byte algorithms under the projected-memory64MiB guard."""
import argparse,json,os,sys
from ASR18_FLAC_carrier_guard_v4 import CarrierGuard
from ASR18_FLAC_carrier_guard_v3 import metadata,now,write_json
from prepare_ASR18_KIRAAT_FLAC_records_v3 import prepare,audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    config=metadata(args.config);guard=CarrierGuard(config);code=0
    try:
        result=prepare(config,guard) if config['mode']=='prepare' else audit(config,guard)
        assert not any(name in sys.modules for name in ('torch','soundfile','av','transformers'))
        write_json(guard.run/'result.json',{**result,'mode':config['mode'],'at_utc':now(),
            'PID':os.getpid(),'actual_paced_IO_bytes':guard.bytes,'actual_new_payload_bytes':guard.payload_written})
        print(json.dumps({'PID':os.getpid(),'mode':config['mode'],**result}),flush=True)
    except BaseException as error:
        code=75 if type(error).__name__=='DataStop' else 1
        write_json(guard.run/'failure.json',{'at_utc':now(),'error_class':type(error).__name__,
            'reason':str(error),'prefix_and_originals_preserved':True,'paced_IO_bytes':guard.bytes})
        raise
    finally:guard.close(code)


if __name__=='__main__':main()
