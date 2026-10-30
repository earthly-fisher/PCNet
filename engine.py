from pathlib import Path
import numpy as np
import torch
from PIL import Image


def load_weights(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)


def train_epoch(model, loader, optimizer, criterion, device, warming=False):
    model.train()
    total = 0.0
    for image, label, _ in loader:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad(set_to_none=True)
        if warming:
            loss = criterion(model(image), label)
        else:
            stages, prediction, _ = model(image)
            loss = criterion(stages, prediction, label)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        total += loss.item() * image.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, output=None, warming=False):
    model.eval()
    scores = []
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
    for image, label, names in loader:
        result = model(image.to(device))
        prediction = result[-1] if warming else result[1]
        prediction = prediction.cpu()
        for pred, target, name in zip(prediction, label, names):
            binary, truth = pred > 0.5, target > 0.5
            tp = (binary & truth).sum().item()
            fp = (binary & ~truth).sum().item()
            fn = (~binary & truth).sum().item()
            accuracy = (binary == truth).float().mean().item()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            recall = tp / max(tp + fn, 1)
            scores.append((accuracy, f1, recall))
            if output is not None:
                pixels = (pred[0].clamp(0, 1).numpy() * 255).astype(np.uint8)
                Image.fromarray(pixels).save(output / (Path(name).stem + '.png'))
    return dict(zip(('accuracy', 'f1', 'recall'), np.mean(scores, axis=0).tolist()))
