import os
import argparse

from diffusion_box.core.pipegen import diffuser
from diffusion_box.utils import hf_login


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", default=False, help="Generate all prompts in prompts directory.")
    parser.add_argument("--ds", type=str, default=None, help="range of diffusion prompt dataset to use.")

    parser.add_argument("--file", type=str, default=None, help="Path of prompt queue file to generate images from.")
    parser.add_argument("--text", type=str, default=None, help="Text to generate image from.")

    parser.add_argument("--concept", type=str, action="append", default=[], help="Concepts to load.")
    parser.add_argument("--append", type=str, default="", help="Text to append to prompts.")

    parser.add_argument("--amount", type=int, default=1, help="Number of images to generate.")
    parser.add_argument("--kwargs", type=str, default=None, help="kwargs as dict to pass to diffuser")

    parser.add_argument("--login-only", action="store_true", default=False, help="Set token only and quit.")
    parser.add_argument("--token", type=str, default=None, help="huggingface token, only needs to called once to set.")


    args = parser.parse_args()

    if len(args.concept) == 0 and args.text is not None:
        print("Auto loading concepts... (--text beta feature)")
        print("\t If this fails, try specifying concepts with --concept, or ensure <X> is whitespace separated.")
        # get all X for <X> in text
        concepts = [x for x in args.text.split() if x.startswith("<") and x.endswith(">")]
        # pull all words surrounded by <> and load them as concepts
        args.concept = [word[1:-1] for word in args.text.split(' ') if word.startswith("<") and word.endswith(">")]
        
        print(f"\tconcepts to load: {args.concept}")

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
    api = hf_login(args.token)
    if args.login_only == True:
        return

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
        diffuser(text=args.text, amount=args.amount, concept_repo=args.concept, **args.kwargs)