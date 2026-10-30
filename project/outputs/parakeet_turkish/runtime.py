"""Shared native Windows runtime for the pinned Transformers TDT experiment."""
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForTDT, AutoProcessor
from transformers.loss.loss_tdt import tdt_loss

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
DATA = ROOT / 'work' / 'data'

def corrected_tdt_loss(token_logits, duration_logits, labels, logit_lengths,
                       label_lengths, blank_token_id, durations, **kwargs):
    # Pinned upstream's wrapper converts targets to int32; torch.gather needs int64.
    return tdt_loss(token_logits, duration_logits, labels.to(token_logits.device).long(),
                    logit_lengths.to(token_logits.device).long(),
                    label_lengths.to(token_logits.device).long(), blank_token_id,
                    durations, sigma=kwargs.get('sigma', 0.0),
                    reduction=kwargs.get('reduction', 'mean'))

def load_model(path=None):
    if path is None:
        path = (ROOT / 'work' / 'base_model_path.txt').read_text().strip()
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires the local CUDA GPU.')
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForTDT.from_pretrained(path, dtype=torch.float32,
                                           local_files_only=True).cuda()
    model.loss_function = corrected_tdt_loss
    return model, processor

def read_audio(row):
    audio, sr = sf.read(row['audio_filepath'], dtype='float32')
    if sr != 16000 or audio.ndim != 1:
        raise ValueError('Prepared data must be mono 16 kHz')
    return audio

def inputs_for(processor, row, training=False):
    kwargs = {'audio': [read_audio(row)], 'sampling_rate': 16000, 'return_tensors': 'pt'}
    if training:
        kwargs['text'] = [row['text']]
    inputs = processor(**kwargs)
    return {key: value.cuda() for key, value in inputs.items()}
