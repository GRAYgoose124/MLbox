from super_image import MsrnModel, ImageLoader
from PIL import Image
import requests

# url = "https://paperswithcode.com/media/datasets/Set5-0000002728-07a9793f_zA3bDjj.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

image = Image.open("images/sd2/best.png")

model = MsrnModel.from_pretrained(
    "eugenesiow/msrn-bam", scale=2
)  # scale 2, 3 and 4 models available

inputs = ImageLoader.load_image(image)
preds = model(inputs)

ImageLoader.save_image(
    preds, "./scaled_2x.png"
)  # save the output 2x scaled image to `./scaled_2x.png`
ImageLoader.save_compare(
    inputs, preds, "./scaled_2x_compare.png"
)  # save an output comparing the super-image with a bicubic scaling
