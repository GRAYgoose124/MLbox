from enum import Enum
from pathlib import Path
import requests
import os
import io
import datetime
from PIL import Image

# https://huggingface.co/docs/api-inference/index


class UME(Enum):
    STABLE_DIFFUSION_API_URL = (
        "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    )


def query(payload: dict, api_url: str = UME.STABLE_DIFFUSION_API_URL.value):
    response = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"},
        json=payload,
    )
    return response.content


def image_query(prompt, save=True, show=False, root_dir="diffusions/"):
    image_bytes = query(
        {
            "inputs": prompt,
        }
    )
    image = Image.open(io.BytesIO(image_bytes))

    if show:
        image.show()

    if save:
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        image.save(root_dir / f"img_{datetime.datetime.now()}.png")

    return image_bytes


def main():
    # root path
    image_query("Award winning, stunning, a photo of amazing")


if __name__ == "__main__":
    main()
