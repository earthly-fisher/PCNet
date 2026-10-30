import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data import BinaryDataset
from engine import evaluate, load_weights
from net import PCNet

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Strictly load and evaluate PCNet')
    parser.add_argument('--data', type=Path, default=BASE / 'data')
    parser.add_argument('--checkpoint', type=Path, default=BASE / 'model' / 'best.pth')
    parser.add_argument('--output', type=Path, default=BASE / 'results')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    model = PCNet(1, 1, 5, 0.5, progressive_mode='dual').to(device)
    load_weights(model, args.checkpoint, device)
    loader = DataLoader(BinaryDataset(args.data, 'test'), batch_size=1)
    metrics = evaluate(model, loader, device, args.output)
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
