import os

from itertools import chain

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    GemmaConfig,
    GemmaForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
)

set_seed(42)

print("Torch Version:", torch.__version__)

config = GemmaConfig.from_pretrained(
    "google/gemma-2b",
    attn_implementation="eager",
)
# config.max_position_embeddings = 128
# config.use_cache = False
config.segment_size = config.max_position_embeddings # Add config

print(config)

pretrained_model = GemmaForCausalLM.from_pretrained(
    "google/gemma-2b", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2b", 
)
pretrained_model.save_pretrained('./models/gemma-2b')
config.save_pretrained('./models/gemma-2b')
tokenizer.save_pretrained('./models/gemma-2b')