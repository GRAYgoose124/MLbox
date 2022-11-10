import os
import torch

from diffusion_box.concept import load_concept
from diffusion_box.utils import hf_login, save_prompt


def get_diffusion_pipe(concept_repo=None, model_id="CompVis/stable-diffusion-v1-4", nsfw=False):
    from diffusers import StableDiffusionPipeline

    if concept_repo is not None:
        text_encoder, tokenizer = load_concept(model_id=model_id, repo_id=concept_repo)

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


def diffuser(text=None, prompts_file=None, dataset=None, bounds=(0, 100), concept_repo=None, append="", **kwargs):
    if prompts_file is not None:
        f = open(prompts_file, "r")
    elif dataset is not None:
        from datasets import load_dataset

        dataset = load_dataset(dataset)
        f = dataset['train']['Prompt'][bounds[0]:bounds[1]]
    elif text:
        f = [text]

    pipe = get_diffusion_pipe(concept_repo=concept_repo, nsfw=True)

    all_images = {}
    for line in f:
        line = line + append
        print(f"Prompt: {line}")

        image = pipe(line, **kwargs).images[0]

        save_file = save_prompt(line, image)
        all_images[save_file] = (line, image)

    all_images