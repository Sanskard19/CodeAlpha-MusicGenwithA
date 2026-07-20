import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import MusicLSTM


class SequenceDataset(Dataset):
    def __init__(self, x_path: Path, y_path: Path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            loss = criterion(logits[:, -1, :], yb)
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
    return total_loss / max(1, total_items)


def main():
    parser = argparse.ArgumentParser(description='Train the LSTM music model.')
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab['token2idx'])

    dataset = SequenceDataset(data_dir / 'X.npy', data_dir / 'y.npy')
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MusicLSTM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_items = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits[:, -1, :], yb)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            running_loss += loss.item() * batch_size
            total_items += batch_size

        train_loss = running_loss / max(1, total_items)
        val_loss = evaluate(model, val_loader, criterion, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'vocab_size': vocab_size,
                'embed_dim': args.embed_dim,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'dropout': args.dropout,
            },
            'history': history,
        }
        torch.save(checkpoint, models_dir / f'checkpoint_epoch_{epoch}.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, models_dir / 'best_model.pt')

    with open(models_dir / 'training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.tight_layout()
    plt.savefig(models_dir / 'loss_curve.png', dpi=160)
    print('Training complete. Best model saved to models/best_model.pt')


if __name__ == '__main__':
    main()
