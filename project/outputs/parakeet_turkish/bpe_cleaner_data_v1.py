"""Fixed existing 352,768-clip cleaner pool with its already complete BPE labels.

No new source selection, text rewrite, target cap or audio cache is introduced.
The cleaner phase has its own exact full-population target denominator.
"""
import json,sqlite3
from pathlib import Path
from expanded_training_rows_v1 import reopen_approved_expanded_TR_rows
from bpe_training_label_catalog_v1 import BPETrainingLabels
from bpe_broad_audio_v1 import record,require

PROJECT=Path(__file__).resolve().parents[2];ROOT=Path(r'E:\gpu_training_scratchpad')
SOURCE=PROJECT/'work/data/TR_expanded_WorldSpeech_index_v1'
LABELS=ROOT/'indexes/BPE_full_decoder_expanded_labels_v1'

class CleanerRows:
    def __init__(self):
        source=SOURCE/'index_provenance.json';preparation=LABELS/'actual_preparation_receipt.json'
        require(record(source)['sha256']=='11c7aaa24f108876a53f0a53e1bed45fabb0d0781486435e1be37a2f862c3be9','Exact existing cleaner source index required')
        require(record(preparation)['sha256']=='2f1cef61ccf14d773b35b2c960750cc87d179a55ea50231865e48b53bafd499b','Complete existing cleaner BPE preparation required')
        prepared=json.loads(preparation.read_text());require(prepared['passed'] and prepared['rows']==352768,'Complete352768 BPE source required')
        require(record(prepared['target_catalog']['path'])==prepared['target_catalog'],'Existing cleaner label provenance changed')
        self.source=reopen_approved_expanded_TR_rows(SOURCE,cache_size=256)
        self.labels=BPETrainingLabels(LABELS,self.source,cache_size=128)
        require(len(self.source)==352768 and self.labels.receipt['rows']==352768,'Fixed cleaner pool cardinality')
        total=self.labels.connection.execute('SELECT COUNT(*),SUM(new_u) FROM labels').fetchone()
        require(total==(352768,11972886) and total[1]==prepared['candidate_target_tokens'],'Full cleaner target-token sum differs')
        self.population=352768;self.total_target_tokens=11972886;self.mean_U=self.total_target_tokens/self.population
        self.identity={'kind':'ASR18_cleaner352768_BPE_catalog_binding_v1','source_catalog':record(source),'BPE_preparation':record(preparation),
            'BPE_catalog_provenance':prepared['target_catalog'],'BPE_database':self.labels.receipt['data'],'rows':self.population,
            'full_target_tokens':self.total_target_tokens,'population_mean_U':self.mean_U,'hours':self.source.receipt['hours'],
            'source_ids_sha256':self.source.ids_sha,'quality_or_language_selection_added':False,'augmentation_phase':'cleaner_data_fine_tuning'}

    def row(self,ordinal):
        require(type(ordinal) is int and 1<=ordinal<=self.population,'Exact cleaner source ordinal required')
        source=self.source[ordinal-1];row=self.labels.rewrite_batch([source])[0]
        require(row['source_id']==source['source_id'] and row['text']==source['text'] and row['language']=='tr','Cleaner BPE target must preserve exact source ID/text')
        return {**row,'cleaner_original_ordinal':ordinal,'augmentation_role':'student_tr'}

    def close(self):self.labels.close();self.source.close()
