"""Train the float32 MNIST MLP baseline.

Usage: python model/train.py [--epochs 6]
Saves checkpoint to model/checkpoints/mlp_float.pt and reports test accuracy.
"""

import argparse
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from net import MnistMLP

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"


def get_loaders(batch_size=128):
    # Keep pixels in [0, 1]; quantization maps this range onto int8 later.
    tf = transforms.ToTensor()
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=512),
    )


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    train_loader, test_loader = get_loaders()
    model = MnistMLP()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        acc = evaluate(model, test_loader)
        print(f"epoch {epoch + 1}/{args.epochs}  test_acc={acc:.4f}")

    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / "mlp_float.pt")
    final = evaluate(model, test_loader)
    print(f"saved checkpoint; final float32 test accuracy: {final:.4f}")


if __name__ == "__main__":
    main()
