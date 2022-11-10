import os
import argparse

from diffusion_box.pipegen import diffuser


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", default=False, help="Generate all prompts in prompts directory.")
    parser.add_argument("--file", type=str, default=None, help="Path of prompt queue file to generate images from.")
    parser.add_argument("--append", type=str, default="", help="Text to append to prompts.")
    parser.add_argument("--text", type=str, default=None, help="Text to generate image from.")
    parser.add_argument("--kwargs", type=str, default=None, help="kwargs as dict to pass to diffuser")
    parser.add_argument("--ds", type=str, default=None, help="range of diffusion prompt dataset to use.")
    parser.add_argument("--concept", type=str, default=None, help="Concept repo url to use for text encoder.")

    args = parser.parse_args()

    if args.ds is not None:
        # args.ds is of form start,end
        args.ds = map(lambda x: int(x, 0), args.ds.split(','))

    # Stable diffusion default config
    kwargs = {
        "height": 512,
        "width": 768,
        "num_inference_steps": 50,
        "guidance_scale": 9.0,
    }

    if args.kwargs is not None:
        args.kwargs = eval(args.kwargs)
    else:
        args.kwargs = kwargs
 
    return args


def main():
    args = argparser()

    if args.ds is not None:
        start, end = args.ds
        diffuser(dataset="Gustavosta/Stable-Diffusion-Prompts", bounds=(start, end), concept_repo=args.concept, append=args.append, **args.kwargs)
    
    if args.file is not None:
        diffuser(prompts_file=args.file, concept_repo=args.concept, append=args.append, **args.kwargs)
    elif args.all == True:
        paths = [os.path.join("prompts", f) for f in os.listdir("prompts") if os.path.isfile(os.path.join("prompts", f)) and not f.startswith("_")]
        for path in paths:
            diffuser(prompts_file=path, concept_repo=args.concept, append=args.append, **args.kwargs)
    
    if args.text:
        diffuser(text=args.text, concept_repo=args.concept, **args.kwargs)