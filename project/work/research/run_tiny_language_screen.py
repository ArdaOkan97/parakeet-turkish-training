r"""Reusable entry point for the verified bounded language-share screen.

Apply E:\gpu_training_scratchpad\use_storage.ps1 before launch. A current
authorized timebox.json is required; output prefixes must be new. This command
does not edit transcripts or admit training data.
"""
import argparse,sys,functools,datetime,json
import whisper_tiny_gpu_screen_v11 as final
runner=final.runner

def safe_analyze(engine,vad,rows,decoded):
    engine.last_annotations=[]
    return final.annotated_analyze(engine,vad,rows,decoded)

if __name__=='__main__':
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--codec-workers',type=int,choices=[4,8],default=8);ap.add_argument('--VAD-group',type=int,choices=[64,128],default=64);options,remaining=ap.parse_known_args();sys.argv=[sys.argv[0]]+remaining
    deadline=datetime.datetime.fromisoformat(json.loads((runner.AREA/'timebox.json').read_text())['deadline_utc']);assert datetime.datetime.now(datetime.timezone.utc)<deadline-datetime.timedelta(minutes=3),'A fresh authorized timebox is required before a new job'
    original=runner.run_stream
    runner.run_stream=functools.partial(original,workers=options.codec_workers,group=options.VAD_group)
    runner.TinyEngine=final.ThresholdCheckedEngine;runner.analyze=safe_analyze;runner.__file__=__file__;runner.main()
