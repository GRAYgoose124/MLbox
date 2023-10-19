import os
from glob2 import glob
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path


ROOT = Path(__file__).parent


def concatenate_files_in_directory(directory, limit=None):
    """Concatenate all .md and .txt files in a given directory into a single string."""
    all_files = glob(os.path.join(directory, "*.md")) + glob(
        os.path.join(directory, "*.txt")
    )
    file_contents = []
    for file_path in all_files:
        with open(file_path, "r") as f:
            try:
                file_contents.append(f.read())
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError: {file_path}")
        if limit is not None and len(file_contents) >= limit:
            break
    return "\n".join(file_contents)


def fine_tune(model_name, train_directory, epochs, save_dir):
    if os.path.exists(save_dir):
        model_name = save_dir

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Create a concatenated dataset from all files
    data = concatenate_files_in_directory(train_directory, limit=6.4e3)
    train_file_path = ROOT / "temp_train_file.txt"
    with open(train_file_path, "w") as f:
        f.write(data)

    # Create dataset from concatenated data
    train_dataset = TextDataset(
        tokenizer=tokenizer, file_path=train_file_path, block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=ROOT / "results",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir=ROOT / "logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # train and save
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass
    finally:
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

    os.remove(train_file_path)


def main():
    # fine tune
    ft_ckp_dir = str(ROOT / "custom_distilgpt2")
    if input("Fine tune? [Y/n] ").lower() != "n":
        directory = "/home/grayson/Documents/mindspace/mindspace/brainsplat/"
        fine_tune("distilgpt2", directory, epochs=16, save_dir=ft_ckp_dir)

    # test the fine tuning by talking to it
    while True:
        try:
            user_input = input(">>> ")
            if user_input == "exit":
                break
        except KeyboardInterrupt:
            break

        tokenizer = AutoTokenizer.from_pretrained(ft_ckp_dir)
        model = AutoModelForCausalLM.from_pretrained(ft_ckp_dir)

        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        atn_mask = torch.ones(
            input_ids.shape, dtype=torch.long, device=input_ids.device
        )

        output = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=atn_mask,
        )

        bot_out = f"Bot: {tokenizer.decode(output[0], skip_special_tokens=True)}\n"

        print(bot_out)


if __name__ == "__main__":
    main()
