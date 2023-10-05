import os
import torch


def load_concepts(tokenizer, text_encoder, repo_list=None):
    new_tokens = []

    for repo_id in repo_list:
        new_tokens.append(load_concept(tokenizer, text_encoder, repo_id))
    
    return new_tokens


def load_concept(tokenizer, text_encoder, repo_id=None):
    library_prefix = "sd-concepts-library"
    if not repo_id.startswith(library_prefix):
        repo_id = f"{library_prefix}/{repo_id}"

    embed_path, token = load_embed(repo_id=repo_id)
    load_learned_embed_in_clip(embed_path, text_encoder, tokenizer)

    return token


# TODO, multiconcepts
def load_embed(repo_id="sd-concepts-library/kuvshinov"):
    from huggingface_hub import hf_hub_download

    #@markdown (Optional) in case you have a `learned_embeds.bin` file and not a `repo_id`, add the path to `learned_embeds.bin` to the `embeds_url` variable 
    embeds_url = "" #Add the URL or path to a learned_embeds.bin file in case you have one
    token = None

    downloaded_embedding_folder = "./downloaded_embedding"
    if not os.path.exists(downloaded_embedding_folder):
        os.mkdir(downloaded_embedding_folder)
        
    if(not embeds_url):
        embeds_path = hf_hub_download(repo_id=repo_id, repo_type="model", filename="learned_embeds.bin")
        token_path = hf_hub_download(repo_id=repo_id, filename="token_identifier.txt")

        # read the token from the file
        with open(token_path, "r") as f:
            token = f.read()
            token = token[1:-1] # remove the brackets

        # use the token to make a subdirectory in the downloaded_embedding_folder
        token_dir = os.path.join(downloaded_embedding_folder, token)
        if not os.path.exists(token_dir):
            os.mkdir(token_dir)
        downloaded_embedding_folder = token_dir
    
        os.system(f"cp {embeds_path} {downloaded_embedding_folder}")
        os.system(f"cp {token_path} {downloaded_embedding_folder}")
    else:
        os.system(f"wget -q -O {downloaded_embedding_folder}/learned_embeds.bin {embeds_url}")

    learned_embeds_path = f"{downloaded_embedding_folder}/learned_embeds.bin"

    return learned_embeds_path, token





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
