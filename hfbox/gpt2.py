# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-mpnet-base-v2')


def main():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    history = ""
    while True:
        if len(history) > 4096:
            history = history.rsplit("\n", 1)[0]

        user_input = input(">>> ")
        if user_input == "exit":
            break
        elif user_input == "!pop":
            if "\n" in history:
                history = history.rsplit("User:", 1)[0]
            continue
        else:
            history += f"User: {user_input}\n"


        input_ids = tokenizer.encode(history, return_tensors="pt")
        atn_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        
        output = model.generate(input_ids, max_new_tokens=256, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id, attention_mask=atn_mask)

        bot_out = f"Bot: {tokenizer.decode(output[0], skip_special_tokens=True)}\n"
        history += bot_out

        print(bot_out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
