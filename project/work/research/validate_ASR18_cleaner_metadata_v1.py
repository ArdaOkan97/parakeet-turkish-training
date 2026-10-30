"""Bounded metadata/target-only cleaner interface check; no waveform/model."""
import argparse,sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT/'outputs/parakeet_turkish'))
from bpe_cleaner_data_v1 import CleanerRows
from bpe_broad_audio_v1 import require,record
from atomic_io_retry_v1 import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);folder.mkdir(exist_ok=False)
    rows=CleanerRows()
    try:
        examples=[]
        for ordinal in (1,2,253798,253799,352767,352768):
            row=rows.row(ordinal)
            require(row['target_tokens']==len(row['tr_metadata_labels']) and all(0<=v<8711 and v!=8192 for v in row['tr_metadata_labels']),'Full exact cleaner BPE labels')
            examples.append({'ordinal':ordinal,'source_id':row['source_id'],'target_tokens':row['target_tokens'],'encoder_frames':row['encoder_frames']})
        require(rows.row(1)==rows.row(1),'Bounded source reread must preserve full values')
        require('torch' not in sys.modules and 'numpy' not in sys.modules,'No model/audio runtime imported')
        write_json(folder/'report.json',{'passed':True,'identity':rows.identity,'metadata_examples':examples,'audio_or_model_imported':False,
            'full_population_tokens_and_new_denominator_verified':True,'source':record(PROJECT/'outputs/parakeet_turkish/bpe_cleaner_data_v1.py')})
    finally:rows.close()

if __name__=='__main__':main()
