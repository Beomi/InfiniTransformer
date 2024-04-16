import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # TODO: set the GPU device
os.environ["WANDB_PROJECT"] = "InfiniTransformer"
os.environ["WANDB_MODE"] = "offline"


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
print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = "cuda:0"  # set GPU device using CUDA_VISIBLE_DEVICES
else:
    device = "cpu"

config = GemmaConfig.from_pretrained(
    "google/gemma-2b",
    attn_implementation="eager",
)
config.memory_size = config.max_position_embeddings
config.use_cache = False
config.segment_size = config.max_position_embeddings

print(config)

# Create the Gemma model with Infini-attention
model = GemmaForCausalLM(config)
# model = model.from_pretrained("google/gemma-2b")
pretrained_model = GemmaForCausalLM.from_pretrained("google/gemma-2b")
# Step 4: Transfer weights
# Note: This is a simplified example; you need to ensure that each parameter's dimensions match.
for param in model.named_parameters():
    name = param[0]
    if name in pretrained_model.state_dict():
        # Check if dimensions match, and only then assign the weights
        if param[1].size() == pretrained_model.state_dict()[name].size():
            param[1].data = pretrained_model.state_dict()[name].data.clone()
        else:
            print(f"Skipping {name} due to size mismatch.")
print(model)
model = model.to(torch.bfloat16)
model = model.to(device)


wiki = load_dataset("wikitext", "wikitext-2-raw-v1")

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")


def tokenize_function(examples):
    return tokenizer(examples["text"])


column_names = list(wiki["train"].features)
tokenized_datasets = wiki.map(
    tokenize_function, remove_columns=column_names, batched=True
)


block_size = config.segment_size * 4  # to test segment size
print("block_size:", block_size)


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
)

print(lm_datasets)
# print(lm_datasets["train"]["input_ids"][0])

training_args = TrainingArguments(
    output_dir="./models/gemma-2b-wikitext",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,  # to test batch dim
    save_steps=10_000,
    save_total_limit=1,
    report_to="wandb",  # "none" if you don't want to report to wandb
    run_name="gemma-2b-wikitext",
    optim="adafactor",
    learning_rate=5e-5,
    bf16=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    data_collator=default_data_collator,
)

trainer.train()
