import torch
from diffusion_box.utils import hf_login, save_prompt


def get_diffusion_pipe(model_id="CompVis/stable-diffusion-v1-4", nsfw=False):
    from diffusers import StableDiffusionPipeline

    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16)  

    # Disabling NSFW filter on Stable Diffusion
    if nsfw:
        def dummy(images, **kwargs): return images, False 
        pipe.safety_checker = dummy

    pipe = pipe.to("cuda")

    return pipe


def diffuser(prompts=None):
    pipe = get_diffusion_pipe(nsfw=True)

    all_images = {}
    with open(prompts, "r") as f:
        for line in f:
            print(f"Prompt: {line}")

            image = pipe(line, height=512, width=512).images[0]
            save_file = save_prompt(line, image)

            all_images[save_file] = (line, image)

    all_images