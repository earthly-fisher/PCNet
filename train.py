import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from data import BinaryDataset
from engine import evaluate, load_weights, train_epoch
from loss import IGLoss, MultiLoss
from net import PCNet

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='PCNet warming, training and validation')
    parser.add_argument('--data', type=Path, default=BASE / 'data')
    parser.add_argument('--output', type=Path, default=BASE / 'runs')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warming-epochs', type=int, default=10)
    parser.add_argument('--warming-checkpoint', type=Path)
    parser.add_argument('--init', type=Path, help='Strictly load a complete model before training')
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.002)
    args = parser.parse_args()
    if args.epochs < 1 or args.warming_epochs < 0 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs, batch-size and lr must be positive; warming-epochs must be nonnegative')
    if args.init and args.warming_checkpoint:
        parser.error('--init and --warming-checkpoint are mutually exclusive')
    device = torch.device(args.device)
    train_loader = DataLoader(BinaryDataset(args.data, 'train'), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(BinaryDataset(args.data, 'validation'), batch_size=1)
    args.output.mkdir(parents=True, exist_ok=True)
    model = PCNet(1, 1, 5, 0.5, progressive_mode='dual').to(device)
    if args.init:
        load_weights(model, args.init, device)
    elif args.warming_checkpoint:
        load_weights(model.module1, args.warming_checkpoint, device)
    elif args.warming_epochs:
        optimizer = torch.optim.AdamW(model.module1.parameters(), lr=0.01)
        best = -1.0
        for epoch in range(args.warming_epochs):
            loss = train_epoch(model.module1, train_loader, optimizer, IGLoss(), device, warming=True)
            metrics = evaluate(model.module1, val_loader, device, warming=True)
            print(f'Warming {epoch + 1}: loss={loss:.6f} {metrics}', flush=True)
            if metrics['f1'] > best:
                best = metrics['f1']
                torch.save(model.module1.state_dict(), args.output / 'warming.pth')
        load_weights(model.module1, args.output / 'warming.pth', device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, history = -1.0, []
    for epoch in range(args.epochs):
        for group in optimizer.param_groups:
            group['lr'] = args.lr * (1 - epoch / max(80, args.epochs)) ** 0.9
        loss = train_epoch(model, train_loader, optimizer, MultiLoss(), device)
        metrics = evaluate(model, val_loader, device)
        history.append(dict(epoch=epoch + 1, loss=loss, **metrics))
        print(f'Train {epoch + 1}: loss={loss:.6f} {metrics}', flush=True)
        torch.save(model.state_dict(), args.output / 'last.pth')
        if metrics['f1'] > best:
            best = metrics['f1']
            torch.save(model.state_dict(), args.output / 'best.pth')
        (args.output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
