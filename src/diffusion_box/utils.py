import os
import datetime
from PIL import Image


def hf_login(token=None):
    """Login to huggingface.co"""
    # import the relavant libraries for loggin in
    from huggingface_hub import HfApi, HfFolder

    api = HfApi()
    if token is not None:
        api.set_access_token(token)
        HfFolder.save_token(token)

        print(f"Token saved, you do not need to login again.")

    print(api.whoami()['name'])

    return api


def image_grid(imgs):
    "Take a flat list of images and pack them into a grid."
    amount = len(imgs)
    rows = int(amount ** 0.5)
    cols = int(amount / rows)

    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def save_prompt(prompt, image):
    outdir = os.getcwd() + "/output"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # generate filename from prompt, filter invalid path chars and strip whitespace
    filename = "".join([c for c in prompt if c.isalpha() or c.isdigit() or c == " "]).rstrip()
    filename = f"{filename.replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # save the image to the directory
    image.save(f"{outdir}/{filename}.png")
        
