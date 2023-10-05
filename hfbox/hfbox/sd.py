# https://huggingface.co/docs/diffusers/using-diffusers/loading
import datetime
from pathlib import Path
from diffusers import StableDiffusionPipeline
from numpy import sin
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


# disable nsfw filter
def dummy(images, **kwargs):
    return images, [False] * len(images)


pipe.safety_checker = dummy


prompt = "Sexy Beautiful 3D"

# image folder management
img_dir = "images"
project_dir = f"{datetime.datetime.now()}"
for dir in Path(img_dir).iterdir():
    if not list(dir.iterdir()):
        dir.rmdir()
Path(f"{img_dir}/{project_dir}").mkdir(parents=True, exist_ok=True)


# generator = torch.Generator("cuda").manual_seed(0)
for i in range(1, 1000):
    image = pipe(
        prompt,
        num_inference_steps=75,
        guidance_scale=7.5,
        height=512,
        width=512,
        # generator=generator,
    ).images[0]
    image.save(f"{img_dir}/{project_dir}/{i}.png")

# # make video
# import subprocess

# subprocess.run(
#     [
#         "ffmpeg",
#         "-framerate",
#         "10",
#         "-i",
#         f"{img_dir}/{project_dir}/dff_img_%d.png",
#         "-c:v",
#         "libx264",
#         "-profile:v",
#         "high",
#         "-crf",
#         "20",
#         "-pix_fmt",
#         "yuv420p",
#         f"{img_dir}/{project_dir}/dff_video.mp4",
#     ]
# )


# # save 100 samples with slight variation
# for i in range(100):
#     image = pipe(prompt).images[0]
#     image.save(f"images/astronaut_rides_horse_{i}.png")
