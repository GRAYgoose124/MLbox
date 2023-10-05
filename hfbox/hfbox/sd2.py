from diffusers import DiffusionPipeline
import torch
import datetime

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
# base.to("cuda")
base.enable_model_cpu_offload()


while True:
    prompt = "A hyper-realistic Octane render of the beautiful psychedelic fractal universe, with scales from atoms to galaxies, people to planets, and everything in between. 4k, award-winning, and made with love. Enjoy!"
    image = base(prompt=prompt, num_inference_steps=75).images[0]
    image.save(f"images/sd2/{datetime.datetime.now()}.png")


# run both experts
# image = base(
#     prompt=prompt,
#     num_inference_steps=5,
#     denoising_end=0.8,
#     # output_type="latent",
# ).images
# refiner = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0",
#     text_encoder_2=base.text_encoder_2,
#     vae=base.vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# )
# refiner.enable_model_cpu_offload()
# image = refiner(
#     prompt=prompt,
#     num_inference_steps=n_steps,
#     denoising_start=0.8,
#     image=image,
# ).images[0]


# Lets make a script that allows us to run base or refiner by just giving it a text prompt or an image
# import click
# from PIL import Image
# from pathlib import Path


# @click.command()
# @click.option("--prompt", default=None, help="Text prompt to generate image from")
# @click.option("--image", default=None, help="Image to generate image from")
# def main(prompt, image):
#     """


#     Demo usage:

#     python hfbox/sd2.py --prompt "A majestic lion jumping from a big stone at night"

#     python hfbox/sd2.py --image "path/to/image.png"

#     """
#     if prompt:
#         base = DiffusionPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-base-1.0",
#             torch_dtype=torch.float16,
#             variant="fp16",
#             use_safetensors=True,
#         )
#         base.enable_model_cpu_offload()

#         image = base(prompt=prompt, num_inference_steps=75).images[0]
#     elif image:
#         image = Image.open(image)

#         refiner = DiffusionPipeline.from_pretrained(
#             "stabilityai/stable-diffusion-xl-refiner-1.0",
#             text_encoder_2=base.text_encoder_2,
#             vae=base.vae,
#             torch_dtype=torch.float16,
#             use_safetensors=True,
#             variant="fp16",
#         )
#         refiner.enable_model_cpu_offload()

#         image = refiner(
#             prompt=prompt,
#             num_inference_steps=75,
#             denoising_start=0.8,
#             image=image,
#         ).images[0]
#     else:
#         raise ValueError("Please provide either a prompt or an image")
#     image.save(f"{datetime.datetime.now()}.png")


# if __name__ == "__main__":
#     main()
