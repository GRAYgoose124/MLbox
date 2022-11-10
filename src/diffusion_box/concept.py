import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer


def load_concept(model_id="CompVis/stable-diffusion-v1-4", repo_id="sd-concepts-library/cat-toy"):
    embed_path = load_embed(repo_id=repo_id)
    tokenizer, text_encoder = init_clip(model_id=model_id)

    load_learned_embed_in_clip(embed_path, text_encoder, tokenizer)

    return text_encoder, tokenizer


def load_embed(repo_id="sd-concepts-library/kuvshinov"):
    from huggingface_hub import hf_hub_download

    #@markdown (Optional) in case you have a `learned_embeds.bin` file and not a `repo_id`, add the path to `learned_embeds.bin` to the `embeds_url` variable 
    embeds_url = "" #Add the URL or path to a learned_embeds.bin file in case you have one
    placeholder_token_string = "" #Add what is the token string in case you are uploading your own embed

    downloaded_embedding_folder = "./downloaded_embedding"
    if not os.path.exists(downloaded_embedding_folder):
        os.mkdir(downloaded_embedding_folder)
        
    if(not embeds_url):
        embeds_path = hf_hub_download(repo_id=repo_id, filename="learned_embeds.bin")
        token_path = hf_hub_download(repo_id=repo_id, filename="token_identifier.txt")
        os.system(f"cp {embeds_path} {downloaded_embedding_folder}")
        os.system(f"cp {token_path} {downloaded_embedding_folder}")

        with open(f'{downloaded_embedding_folder}/token_identifier.txt', 'r') as file:
            placeholder_token_string = file.read()
    else:
        os.system(f"wget -q -O {downloaded_embedding_folder}/learned_embeds.bin {embeds_url}")

    learned_embeds_path = f"{downloaded_embedding_folder}/learned_embeds.bin"

    return learned_embeds_path


def init_clip(model_id):
    #@title Set up the Tokenizer and the Text Encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    )

    return tokenizer, text_encoder


def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")

    # separate token and the embeds
    trained_token = list(loaded_learned_embeds.keys())[0]
    embeds = loaded_learned_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
