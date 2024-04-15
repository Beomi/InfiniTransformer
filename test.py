import torch
from torch.nn import functional as F
from transformers import GemmaConfig, GemmaForCausalLM, AutoTokenizer, pipeline

config = GemmaConfig.from_pretrained(
    "google/gemma-2b",
    attn_implementation="eager",
)
config.memory_size = 2048
config.use_cache = False
config.segment_size = 16

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

# Generate some dummy input data
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
text = """This work introduces an efficient method to scale Transformer-based Large Language Models (LLMs) to infinitely long inputs with bounded memory and computation. A key component in our proposed approach is a new attention technique dubbed Infini-attention. The Infini-attention incorporates a compressive memory into the vanilla attention mechanism and builds in both masked local attention and long-term linear attention mechanisms in a single Transformer block. We demonstrate the effectiveness of our approach on long-context language modeling benchmarks, 1M sequence length passkey context block retrieval and 500K length book summarization tasks with 1B and 8B LLMs. Our approach introduces minimal bounded memory parameters and enables fast streaming inference for"""
encoded = tokenizer(
    text,
    return_tensors="pt",
)
# attention_mask = torch.ones_like(input_ids)
encoded["labels"] = encoded["input_ids"].clone()

print(encoded)
# Test the forward pass
outputs = model(**encoded)  # position_ids=position_ids)
print(outputs.loss)

outputs.loss.backward()  # Test the backward pass

print("backprop done")


# generated = model.generate(
#     **encoded,
#     max_new_tokens=10,
#     do_sample=True,
#     num_return_sequences=1,
# )
# print(tokenizer.decode(generated[0], skip_special_tokens=False))


# Step 1: Get effective batch size and sequence length
batch_size = encoded["input_ids"].shape[0]
sequence_length = encoded["input_ids"].shape[1]

# Step 2: Prepare input data for generation
input_ids = encoded["input_ids"]
attention_mask = encoded.get("attention_mask", None)

# Step 3: Initialize past
past = None

# Step 4: Start generation loop
for _ in range(10):  # 10 is the number of new tokens to generate
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
print(generated_sequence)
