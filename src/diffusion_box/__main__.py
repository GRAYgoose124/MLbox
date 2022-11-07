import torch
import os
import matplotlib.pyplot as plt

from diffusion_box.utils import image_grid


def hf_login(token=None):
    """Login to huggingface.co"""
    # import the relavant libraries for loggin in
    from huggingface_hub import HfApi, HfFolder

    api = HfApi()
    if token is not None:
        api.set_access_token(token)
        HfFolder.save_token(token)
    else:
        token = HfFolder.get_token()

    api.whoami()
    return api


def get_diffusion_pipe(nsfw=False):
    from diffusers import StableDiffusionPipeline

    # make sure you're logged in with `huggingface-cli login`
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)  

    # Disabling NSFW filter on Stable Diffusion
    if nsfw:
        def dummy(images, **kwargs): return images, False 
        pipe.safety_checker = dummy

    pipe = pipe.to("cuda")

    return pipe


def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="text to prompt, interactive mode by default")
    parser.add_argument("--show", action="store_true", default=False, help="show the images")
    parser.add_argument("--token", type=str, default=None, help="huggingface token, only needs to called once to set.")

    return parser.parse_args()


def save_images(images):
    last_index = 0
    directory = os.getcwd() + "/output/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        last_index = len(os.listdir(directory))

    # save the image to the directory
    for i, image in enumerate(images):
        image.save(f"{directory}/{i + last_index}.png")


def prompt(text, amount=1, show=False, pipe=None):
    """Prompt the model with a text"""
    if pipe is None:
        pipe = get_diffusion_pipe(nsfw=True)

    images = pipe([text] * amount).images 

    if show:
        # divide the images into a grid rows x cols
        # make rows and cols as close as possible
        rows = int(amount ** 0.5)
        cols = int(amount / rows)
        grid = image_grid(images, rows, cols)
        plt.imshow(grid)

    save_images(images)


def interactive_prompt(show=False, pipe=None):
    # Avoid 
    if pipe is None:
        pipe = get_diffusion_pipe(nsfw=True)

    try:
        text = ""
        while text != "exit":
            text = input("Enter a prompt: ")
            try:
                amount = int(input("Enter the amount of images to generate: "))
            except ValueError:
                amount = 1

            if text == "exit":
                break
            
            prompt(text=text, amount=amount, show=show, pipe=pipe)
    except KeyboardInterrupt:
        pass


def main():
    args = argparser()

    api = hf_login(args.token)

    if args.text is None:
        interactive_prompt(show=args.show)
    else:
        prompt(args.text, show=args.show)


if __name__ == '__main__':
    main()  