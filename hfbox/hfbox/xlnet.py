from transformers import XLNetTokenizer, XLNetModel

xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
xlnet_model = XLNetModel.from_pretrained("xlnet-large-cased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
