from diffusers import DiffusionPipeline
import torch
import datetime
from PIL import Image

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

# get python version
import sys

if sys.version_info[0] < 3 and sys.version_info[1] == 9:
    refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
refiner.enable_model_cpu_offload()

image = Image.open("images/sd2/best.png")


while True:
    prompt = "A hyper-realistic Octane render of the beautiful psychedelic fractal universe, with scales from atoms to galaxies, people to planets, and everything in between. 4k, award-winning, and made with love. Enjoy!"
    image = refiner(prompt=prompt, num_inference_steps=75, image=image).images[0]
    image.save(f"images/sd2/anim/{datetime.datetime.now()}.png")
