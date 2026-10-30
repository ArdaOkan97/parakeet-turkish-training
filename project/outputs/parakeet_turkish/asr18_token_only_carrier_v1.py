"""Native training vocabulary carrier; precomputed audio inputs are mandatory."""
import json
from pathlib import Path


class TokenOnlyCarrier:
    def __init__(self,tokenizer,blank_token):
        self.tokenizer=tokenizer;self.blank_token=blank_token
        if self.blank_token_id!=8192:
            raise ValueError('Original immutable blank token ID8192 required')
    @property
    def blank_token_id(self):return self.tokenizer.convert_tokens_to_ids(self.blank_token)
    def __call__(self,*args,**kwargs):
        raise ValueError('Audio frontend unavailable in native training process; verified precomputed inputs required')


def load(base):
    from transformers import AutoTokenizer
    base=Path(base);configuration=base/'processor_config.json'
    saved=json.loads(configuration.read_text()) if configuration.exists() else {}
    tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
    return TokenOnlyCarrier(tokenizer,saved.get('blank_token','<blank>'))
