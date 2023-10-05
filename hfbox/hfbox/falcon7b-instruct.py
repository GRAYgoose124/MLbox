import torch
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

# force free gpu mem


# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Model and Tokenizer loading
model_id = "vilsonrodrigues/falcon-7b-instruct-sharded"
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Inference pipeline setup
text_gen_pipeline = pipeline(
    "text-generation",
    model=model_4bit,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=2048,
    do_sample=True,
    top_k=10,
    num_return_sequences=10,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# Sample Inference

rolling_prompt = ""
user_input = ""
while user_input != "EXIT":
    user_input = input("User: ")
    rolling_prompt += f"User: {user_input}\nBot: "
    sequences = text_gen_pipeline(rolling_prompt)

    for i, alt in enumerate(sequences):
        print(f"\t{i}: {alt['generated_text']} {'DEFAULT' if i == 0 else ''}")

    try:
        alt = int(input("Select an alt if better: "))
    except:
        alt = 0

    rolling_prompt += sequences[alt]["generated_text"]
    print(rolling_prompt)
