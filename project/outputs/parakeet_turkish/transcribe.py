"""Transcribe local audio with the selected pilot and measured confidence.

Usage: python transcribe.py recording.wav --language tr
Language controls confidence calibration and text normalization only. It does
not prompt or force the ASR decoder's language.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from runtime import OUTPUT, load_model
from confidence import generate_with_confidence

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio',type=Path)
    parser.add_argument('--language',choices=['tr','en','es','it','fr','de'],default='tr')
    parser.add_argument('--model-path',type=Path,default=OUTPUT/'checkpoint')
    parser.add_argument('--calibration-path',type=Path,
                        help='Development-fitted calibration for this exact model. The selected checkpoint uses its saved calibration automatically.')
    args=parser.parse_args()
    if args.calibration_path is not None and not args.calibration_path.is_file():
        parser.error('--calibration-path must point to an existing JSON file')
    model,processor=load_model(args.model_path);model.eval()
    signal,sr=sf.read(args.audio,dtype='float32',always_2d=True)
    signal=signal.mean(axis=1)
    if sr!=16000:
        divisor=math.gcd(sr,16000);signal=resample_poly(signal,16000//divisor,sr//divisor)
    inputs=processor(audio=[signal],sampling_rate=16000,return_tensors='pt')
    inputs={key:value.cuda() for key,value in inputs.items()}
    calibration_path=args.calibration_path
    if calibration_path is None and args.model_path.resolve()==(OUTPUT/'checkpoint').resolve():
        calibration_path=OUTPUT/'confidence_calibration.json'
    calibration=(json.loads(calibration_path.read_text(encoding='utf-8'))
                 if calibration_path is not None and calibration_path.exists() else {})
    temperature=calibration.get(args.language,{}).get('temperature',1.0)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        generated,confidence=generate_with_confidence(model,processor.tokenizer,inputs,
            language=args.language,temperatures=[1.0,temperature])
    key=format(float(temperature),'.8g')
    result={'transcript':processor.decode(generated.sequences[0],skip_special_tokens=True),
            'confidence_temperature':temperature,'confidence_language':args.language,
            'words':[{'word':word['surface'],'raw_confidence':word['confidence'],
                      'calibrated_confidence':word['confidence_by_temperature'][key],
                      'conditional_nonblank_confidence':word['conditional_nonblank_confidence']}
                     for word in confidence['words']]}
    print(json.dumps(result,ensure_ascii=False,indent=2))
