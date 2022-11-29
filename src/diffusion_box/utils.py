import json
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
    # index_path = outdir + "/index.json"

    # if not os.path.exists(outdir):
    #     os.mkdir(outdir)
    #     with open(index_path, "w") as f:
    #         json.dump({}, f)

    try:
        letters = "".join(word[0] for word in prompt.split(' '))
    except IndexError:
        letters = ""
    filename = f"{outdir}/{letters}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
  
    # Save the image
    image.save(filename)

    # # Save the prompt and filename to the index.json
    # with open(index_path, "a+") as f:
    #     try:
    #         index = json.load(f)
    #     except json.decoder.JSONDecodeError:
    #         index = {}

    #     if prompt not in index:
    #         index[prompt] = [filename]
    #     else:
    #         index[prompt].append(filename)

    #     json.dump(index, f)

        
