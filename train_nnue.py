import os
import struct
import random
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader


# ============================================================
# SimpleLogics NNUE
#
# Independent network architecture.
#
# Input:
#   12 piece types x 64 squares
#
# Network:
#   768 -> 256 -> 32 -> 1
#
# Quantized export:
#   int16 weights
# ============================================================

INPUTS = 12 * 64
HIDDEN1 = 256
HIDDEN2 = 32

SCALE = 256.0


class SimpleLogicsNNUE(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(INPUTS, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.out = nn.Linear(HIDDEN2, 1)

    def forward(self, x):

        x = torch.clamp(self.fc1(x), 0.0, 1.0)
        x = torch.clamp(self.fc2(x), 0.0, 1.0)

        return self.out(x)


# ============================================================
# FEN -> feature vector
# ============================================================

PIECE_INDEX = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def fen_features(fen):

    board = fen.split()[0]

    x = [0.0] * INPUTS

    square = 0

    for ch in board:

        if ch == "/":
            continue

        if ch.isdigit():
            square += int(ch)
            continue

        if ch in PIECE_INDEX:

            idx = PIECE_INDEX[ch]

            if square < 64:
                x[idx * 64 + square] = 1.0

            square += 1

    return x


# ============================================================
# Dataset
#
# Expected format:
#
# FEN | SCORE
#
# Example:
#
# rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0.0
#
# score is centipawns.
# ============================================================

class PositionDataset(IterableDataset):

    def __init__(self, files, limit=None):

        self.files = files
        self.limit = limit

    def parse_line(self, line):

        line = line.strip()

        if not line:
            return None

        if "|" not in line:
            return None

        fen, score = line.split("|", 1)

        fen = fen.strip()
        score = score.strip()

        try:
            score = float(score)
        except:
            return None

        # normalize centipawn score
        score = max(-1000.0, min(1000.0, score))
        score /= 1000.0

        try:
            features = fen_features(fen)
        except:
            return None

        return features, score

    def __iter__(self):

        count = 0

        files = list(self.files)
        random.shuffle(files)

        for filename in files:

            try:
                f = open(filename, "r", encoding="utf-8")
            except:
                continue

            with f:

                for line in f:

                    item = self.parse_line(line)

                    if item is None:
                        continue

                    yield item

                    count += 1

                    if self.limit and count >= self.limit:
                        return


# ============================================================
# Validation
# ============================================================

def validate(model, loader, device):

    model.eval()

    total = 0.0
    count = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)

            pred = model(x)

            loss = torch.mean((pred - y) ** 2)

            total += loss.item()
            count += 1

            if count >= 100:
                break

    model.train()

    if count == 0:
        return 999.0

    return total / count


# ============================================================
# Export binary NNUE
# ============================================================

def export_nnue(model, filename):

    model.eval()

    fc1_w = model.fc1.weight.detach().cpu()
    fc1_b = model.fc1.bias.detach().cpu()

    fc2_w = model.fc2.weight.detach().cpu()
    fc2_b = model.fc2.bias.detach().cpu()

    out_w = model.out.weight.detach().cpu()
    out_b = model.out.bias.detach().cpu()

    with open(filename, "wb") as f:

        # Magic
        f.write(b"SLNNUE2")

        # Architecture
        f.write(
            struct.pack(
                "<IIII",
                INPUTS,
                HIDDEN1,
                HIDDEN2,
                1
            )
        )

        def write_tensor(t):

            values = t.flatten().tolist()

            q = []

            for v in values:

                v = max(-32767.0, min(32767.0, float(v) * SCALE))

                q.append(int(round(v)))

            f.write(struct.pack("<" + "h" * len(q), *q))

        write_tensor(fc1_w)
        write_tensor(fc1_b)

        write_tensor(fc2_w)
        write_tensor(fc2_b)

        write_tensor(out_w)
        write_tensor(out_b)

    print("NNUE exported:", filename)


# ============================================================
# Training
# ============================================================

def train(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu
        else "cpu"
    )

    print("Device:", device)

    model = SimpleLogicsNNUE().to(device)

    if args.resume and os.path.exists(args.resume):

        print("Loading:", args.resume)

        model.load_state_dict(
            torch.load(
                args.resume,
                map_location=device
            )
        )

    dataset = PositionDataset(
        args.data,
        limit=args.limit
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    loss_fn = nn.SmoothL1Loss()

    model.train()

    step = 0

    for epoch in range(args.epochs):

        print("Epoch:", epoch + 1)

        for x, y in loader:

            x = x.to(device).float()
            y = y.to(device).float().unsqueeze(1)

            optimizer.zero_grad()

            prediction = model(x)

            loss = loss_fn(
                prediction,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            step += 1

            if step % 100 == 0:

                print(
                    "step",
                    step,
                    "loss",
                    float(loss)
                )

            if step % args.save_every == 0:

                torch.save(
                    model.state_dict(),
                    args.checkpoint
                )

                print(
                    "checkpoint:",
                    args.checkpoint
                )

            if args.steps and step >= args.steps:
                break

        if args.steps and step >= args.steps:
            break

    torch.save(
        model.state_dict(),
        args.checkpoint
    )

    export_nnue(
        model,
        args.output
    )

    print("Training complete.")
    print("Checkpoint:", args.checkpoint)
    print("NNUE:", args.output)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--output",
        default="simplelogics.nnue"
    )

    parser.add_argument(
        "--checkpoint",
        default="simplelogics_checkpoint.pt"
    )

    parser.add_argument(
        "--resume",
        default=""
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=0
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10000
    )

    parser.add_argument(
        "--cpu",
        action="store_true"
    )

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
