from diffusion_box.pipegen import diffuser
def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None, help="Path of prompt queue file to generate images from.")

    return parser.parse_args()


def main():
    args = argparser()

    if args.file is None:
        # Get prompt files from $CWD/prompts directory
        import os
        paths = [os.path.join("prompts", f) for f in os.listdir("prompts") if os.path.isfile(os.path.join("prompts", f))]
        # recursively call diffuser on each path. 
        for path in paths:
            diffuser(path)
    else:
        diffuser(args.file)