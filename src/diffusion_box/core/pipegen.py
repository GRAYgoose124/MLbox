import os
from numpy import isin
import torch

from transformers import CLIPTextModel, CLIPTokenizer

from diffusion_box.core.concept import  load_concept
from diffusion_box.utils import hf_login, save_prompt


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

def get_diffusion_pipe(concept_repo=None, model_id="CompVis/stable-diffusion-v1-4", nsfw=False):
    from diffusers import StableDiffusionPipeline
    tokenizer, text_encoder = init_clip(model_id=model_id)
    
    if isinstance(concept_repo, str):
        load_concept(tokenizer, text_encoder, repo_id=concept_repo)
    elif isinstance(concept_repo, list):
        for repo in concept_repo:
            load_concept(tokenizer, text_encoder, repo_id=repo)

    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        revision="fp16", 
        torch_dtype=torch.float16, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer
    ) 

    # Disabling NSFW filter on Stable Diffusion
    if nsfw:
        def dummy(images, **kwargs): return images, False 
        pipe.safety_checker = dummy

    pipe = pipe.to("cuda")

    return pipe


# TODO dynamically load concepts from <> and gen pipe, cache learning embeds, etc.
def diffuser(text=None, amount=1, prompts_file=None, dataset=None, bounds=(0, 100), concept_repo=None, append="", **kwargs):
    if prompts_file is not None:
        f = open(prompts_file, "r")
    elif dataset is not None:
        from datasets import load_dataset

        dataset = load_dataset(dataset)
        f = dataset['train']['Prompt'][bounds[0]:bounds[1]]
    elif text:
        f = [text] * amount

    pipe = get_diffusion_pipe(concept_repo=concept_repo, nsfw=True)

    all_images = {}
    for line in f:
        line = line + append
        print(f"Prompt: {line}")

        image = pipe(line, **kwargs).images[0]

        save_file = save_prompt(line, image)
        all_images[save_file] = (line, image)

    all_images