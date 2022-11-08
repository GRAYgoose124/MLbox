import readline
import json
import os
import matplotlib.pyplot as plt


from diffusion_box.utils import hf_login, image_grid
from diffusion_box.pipegen import get_diffusion_pipe


def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default=None, help="text to prompt, interactive mode by default")
    parser.add_argument("--show", action="store_true", default=False, help="show the images")
    parser.add_argument("--token", type=str, default=None, help="huggingface token, only needs to called once to set.")
    parser.add_argument("--file", type=str, default=None, help="Path of prompt queue file to generate images from.")
    parser.add_argument("--login-only", action="store_true", default=False, help="Set token only and quit.")
    return parser.parse_args()


def prompt(text, amount=1, show=False, pipe=None):
    """Prompt the model with a text"""
    if pipe is None:
        pipe = get_diffusion_pipe(nsfw=True)

    images = []
    # One image at a time, for low-VRAM GPUs :(
    for _ in range(amount):
        images.append(pipe(text, height=512, width=512).images[0])

    if show:
        # divide the images into a grid rows x cols
        # make rows and cols as close as possible

        plt.show(images)

    save_images(text, images)


def interactive_prompt(show=False, pipe=None):
    # Avoid 
    if pipe is None:
        pipe = get_diffusion_pipe(nsfw=True)

    try:
        text = ""
        while text != "exit":
            text = input("Enter a prompt: ")
            if text == "exit":
                break
            
            prompt(text=text, show=show, pipe=pipe)
    except KeyboardInterrupt:
        pass


def main():
    args = argparser()

    api = hf_login(args.token)
    if args.login_only == True:
        return

    if args.file is not None:
        pipe = get_diffusion_pipe(nsfw=True)
        with open(args.file, "r") as f:
            for line in f:
                prompt(line.split("#")[0], show=args.show, pipe=pipe)
                
    elif args.text is None:
        interactive_prompt(show=args.show)
    else:
        prompt(args.text, show=args.show)


if __name__ == '__main__':
    main()  