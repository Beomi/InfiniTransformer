import torch
from transformers import GemmaConfig, GemmaForCausalLM, AutoTokenizer, pipeline

config = GemmaConfig.from_pretrained(
    "google/gemma-2b",
    attn_implementation="eager",
)
config.memory_size = 2048
config.use_cache = False

print(config)

# Create the Gemma model with Infini-attention
model = GemmaForCausalLM(config)
model = model.from_pretrained("google/gemma-2b")
print(model)

# Generate some dummy input data
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
encoded = tokenizer("Hello, how are you?", return_tensors="pt")
# attention_mask = torch.ones_like(input_ids)
encoded["labels"] = encoded["input_ids"].clone()

print(encoded)
# Test the forward pass
outputs = model(**encoded)  # position_ids=position_ids)
print(outputs)

outputs.loss.backward()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("Hello, how are you?", max_new_tokens=50))
