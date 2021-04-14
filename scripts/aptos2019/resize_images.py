from functools import partial

from PIL import Image
from pathlib import Path

from multiprocessing import Pool
import argparse
import os


def resize_one(path, size=(384, 384), output_dir="resized", callback=None):
    output_dir = Path(output_dir)
    image = Image.open(path)
    image = image.resize(size, resample=Image.LANCZOS)
    image.save(output_dir / path.name)
    if callback is not None:
        callback()
    print(output_dir / path.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    f = partial(resize_one, size=(args.size, args.size), output_dir=args.output_dir)
    with Pool(8) as p:
        p.map(f, Path(args.image_dir).glob("*.png"))
