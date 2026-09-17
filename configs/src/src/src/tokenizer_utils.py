"""Tokenizer loading + batch encoding for M3/M4/M5."""
from transformers import AutoTokenizer
from typing import List, Dict

def load_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(name, use_fast=True)

def encode_texts(tokenizer, texts: List[str],
                 max_len: int = 128) -> Dict:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_attention_mask=True,
        return_token_type_ids=False,   # XLM-R / RoBERTa have no token_type_ids
    )
