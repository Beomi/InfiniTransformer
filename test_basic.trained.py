import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # TODO: set the GPU device

import torch
from torch.nn import functional as F
from transformers import GemmaConfig, GemmaForCausalLM, AutoTokenizer, pipeline

model = GemmaForCausalLM.from_pretrained(
    'models/gemma-2b-wikitext/checkpoint-297',
    torch_dtype='auto',
    device_map={'':0},
    attn_implementation="eager",
)

print(model)

# Generate some dummy input data
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
text = """This work introduces an efficient method to scale Transformer-based"""

encoded = tokenizer(
    text,
    return_tensors="pt",
).to(model.device)

# Step 1: Get effective batch size and sequence length
batch_size = encoded["input_ids"].shape[0]
sequence_length = encoded["input_ids"].shape[1]

# Step 2: Prepare input data for generation
input_ids = encoded["input_ids"]
attention_mask = encoded.get("attention_mask", None)

# Step 3: Initialize past
past = None

# Step 4: Start generation loop
for _ in range(500):  # 10 is the number of new tokens to generate
    with torch.no_grad():
        # Get next token scores
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past,
        )
        next_token_logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values

        # Perform sampling to get the next token
        next_token = torch.multinomial(
            F.softmax(next_token_logits, dim=-1), num_samples=1
        )

        # Update input_ids, attention_mask, and past
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = F.pad(attention_mask, (0, 1), value=1)

# Step 5: Return generated sequence
generated_sequence = tokenizer.decode(input_ids[0], skip_special_tokens=False)
print("Input:")
print(text)
print("generated_sequence:")
print(generated_sequence.replace(text, ''))

# Test .generate() method
generated = model.generate(
    **encoded,
    max_new_tokens=1024,
    do_sample=True,
    num_return_sequences=1,
)
print("Generated:")
print(tokenizer.decode(generated[0], skip_special_tokens=False).replace(text, ''))
