from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("abacaj/codegen-16B-nl-sharded")
model = AutoModelForCausalLM.from_pretrained("abacaj/codegen-16B-nl-sharded")

text = "def hello_world():"
input_ids = tokenizer(text, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids, max_length=128)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
