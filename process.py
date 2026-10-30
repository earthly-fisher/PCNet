"""Prepare grayscale segmentation data with offline augmentation."""
import argparse
from pathlib import Path
import random
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent


def prepare(source, output, copies=500, size=512, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Source and output directories must be separate')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be empty to avoid overwriting existing data')
    if copies < 1 or size < 16 or size % 16:
        raise ValueError('copies must be positive; size must be a positive multiple of 16')
    if not (source / "validation" / "images").is_dir():
        raise ValueError("Provide a separate validation/images split before preprocessing")
    rng = random.Random(seed)
    noise_rng = np.random.default_rng(seed)
    counts = {}
    for split, original in [('train', 'training'), ('validation', 'validation'), ('test', 'test')]:
        folder = source / original
        if not folder.exists() and split == 'validation':
            counts[split] = 0
            continue
        images = sorted(x for x in (folder / 'images').iterdir() if x.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'})
        if not images:
            raise ValueError(f'No images in {folder / "images"}')
        for kind in ['image', 'vessel']:
            (output / split / kind).mkdir(parents=True, exist_ok=True)
        count = 0
        for path in images:
            mask_path = folder / 'vessel' / path.name
            if not mask_path.exists():
                mask_path = folder / 'label' / path.name
            with Image.open(path) as im:
                image = im.convert('L').resize((size, size), Image.Resampling.BICUBIC)
            with Image.open(mask_path) as im:
                mask = im.convert('L').resize((size, size), Image.Resampling.NEAREST)
            for index in range(copies if split == 'train' else 1):
                img, lab = image.copy(), mask.copy()
                if split == 'train':
                    if rng.random() >= 0.2:
                        angle = rng.randrange(90)
                        if rng.random() < 0.5:
                            img, lab = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT), lab.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        if rng.random() < 0.5:
                            img, lab = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM), lab.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                        img = img.rotate(angle, resample=Image.Resampling.NEAREST)
                        lab = lab.rotate(angle, resample=Image.Resampling.NEAREST)
                    if rng.random() >= 0.8:
                        variance = noise_rng.choice([0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002])
                        pixels = np.asarray(img, dtype=np.float32) / 255
                        pixels = np.clip(pixels + noise_rng.normal(0, np.sqrt(variance), pixels.shape), 0, 1)
                        img = Image.fromarray(np.rint(pixels * 255).astype(np.uint8))
                lab = Image.fromarray((np.asarray(lab) > 127).astype(np.uint8) * 255)
                name = f'{path.stem}_{path.suffix[1:]}_{index}.png'
                img.save(output / split / 'image' / name)
                lab.save(output / split / 'vessel' / name)
                count += 1
        counts[split] = count
    if not counts['validation']:
        raise ValueError('Provide a separate validation/images and validation/vessel (or label) split; test data is not used for model selection')
    return counts


def main():
    parser = argparse.ArgumentParser(description='Offline rotation, flips and Gaussian noise before training')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=BASE / 'data')
    parser.add_argument('--copies', type=int, default=500)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(prepare(args.source, args.output, args.copies, args.size, args.seed))


if __name__ == '__main__':
    main()
