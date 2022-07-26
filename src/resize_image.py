from PIL import Image
import argparse
from pathlib import Path
import re
import os

from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--num', type=int, default=28000)

args = parser.parse_args()

img_dir = Path(args.input)
img_paths = [p for p in img_dir.iterdir()]

img_paths.sort(key=lambda s: int(re.search(r'(\d+).png', str(s)).group(0).replace('.png', '')))
if len(img_paths) > args.num:
    img_paths = img_paths[:args.num]

if not os.path.isdir(args.output):
    os.makedirs(args.output)

for img_path in tqdm(img_paths):
    img = Image.open(img_path)
    number = re.search(r'(\d+).png', str(img_path)).group(0).replace('.png', '')

    img_resize = img.resize((args.size, args.size), resample = Image.BILINEAR)
    img_resize.save(args.output+'/'+str(number)+'.png')


