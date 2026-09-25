#!/usr/bin/env python3

import argparse
import math
import os
import random
import struct
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# SimpleLogics NNUE
# Architecture:
#   768 -> 256 -> 32 -> 1
#
# Input:
#   12 piece types x 64 squares
#
# Format dataset:
#   FEN | SCORE
#
# SCORE = centipawns from White's point of view
# ============================================================

INPUTS = 768
HIDDEN1 = 256
HIDDEN2 = 32
OUTPUTS = 1

MAGIC = b"SLNNUE2"


# ============================================================
# FEN -> 768 FEATURES
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
    """
    Convert FEN into a 768-element float tensor.

    Square mapping:
        a1 = 0
        b1 = 1
        ...
        h1 = 7
        a2 = 8
        ...
        h8 = 63
    """

    board = fen.split()[0]

    x = [0.0] * INPUTS

    rank = 7
    file = 0

    for ch in board:

        if ch == "/":
            rank -= 1
            file = 0
            continue

        if ch.isdigit():
            file += int(ch)
            continue

        if ch in PIECE_INDEX:
            sq = rank * 8 + file
            piece = PIECE_INDEX[ch]

            index = piece * 64 + sq

            if 0 <= index < INPUTS:
                x[index] = 1.0

            file += 1

    return torch.tensor(x, dtype=torch.float32)


# ============================================================
# DATASET
# ============================================================

class PositionDataset(Dataset):

    def __init__(self, filename, limit=0):

        self.samples = []

        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan: {filename}"
            )

        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                if line.startswith("#"):
                    continue

                if "|" not in line:
                    continue

                fen, score = line.split("|", 1)

                fen = fen.strip()
                score = score.strip()

                try:
                    score = float(score)
                except ValueError:
                    continue

                try:
                    features = fen_features(fen)
                except Exception:
                    continue

                self.samples.append(
                    (
                        features,
                        torch.tensor(
                            [score],
                            dtype=torch.float32
                        )
                    )
                )

                if limit > 0 and len(self.samples) >= limit:
                    break

        if len(self.samples) == 0:
            raise RuntimeError(
                "Dataset kosong atau format FEN | SCORE salah."
            )

        print(
            f"Loaded {len(self.samples):,} positions"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


# ============================================================
# MODEL
# ============================================================

class SimpleLogicsNNUE(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear(
            INPUTS,
            HIDDEN1
        )

        self.fc2 = nn.Linear(
            HIDDEN1,
            HIDDEN2
        )

        self.out = nn.Linear(
            HIDDEN2,
            OUTPUTS
        )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.kaiming_uniform_(
            self.fc1.weight,
            a=math.sqrt(5)
        )

        nn.init.zeros_(
            self.fc1.bias
        )

        nn.init.kaiming_uniform_(
            self.fc2.weight,
            a=math.sqrt(5)
        )

        nn.init.zeros_(
            self.fc2.bias
        )

        nn.init.uniform_(
            self.out.weight,
            -0.01,
            0.01
        )

        nn.init.zeros_(
            self.out.bias
        )

    def forward(self, x):

        x = self.fc1(x)

        # clipped ReLU
        x = torch.clamp(
            x,
            0.0,
            1.0
        )

        x = self.fc2(x)

        x = torch.clamp(
            x,
            0.0,
            1.0
        )

        x = self.out(x)

        return x


# ============================================================
# EXPORT BINARY
# ============================================================

def export_nnue(model, output_file):

    model = model.cpu()

    with torch.no_grad():

        w1 = (
            model.fc1.weight
            .detach()
            .numpy()
        )

        b1 = (
            model.fc1.bias
            .detach()
            .numpy()
        )

        w2 = (
            model.fc2.weight
            .detach()
            .numpy()
        )

        b2 = (
            model.fc2.bias
            .detach()
            .numpy()
        )

        wo = (
            model.out.weight
            .detach()
            .numpy()
        )

        bo = (
            model.out.bias
            .detach()
            .numpy()
        )

    # --------------------------------------------------------
    # Fixed-point scale
    # --------------------------------------------------------

    SCALE = 256.0

    w1_q = [
        max(-32768, min(32767, int(round(v * SCALE))))
        for v in w1.reshape(-1)
    ]

    b1_q = [
        max(-32768, min(32767, int(round(v * SCALE))))
        for v in b1.reshape(-1)
    ]

    w2_q = [
        max(-32768, min(32767, int(round(v * SCALE))))
        for v in w2.reshape(-1)
    ]

    b2_q = [
        max(-32768, min(32767, int(round(v * SCALE))))
        for v in b2.reshape(-1)
    ]

    wo_q = [
        max(-32768, min(32767, int(round(v * SCALE))))
        for v in wo.reshape(-1)
    ]

    bo_q = [
        max(-32768, min(32767, int(round(v * SCALE))))
        for v in bo.reshape(-1)
    ]

    with open(output_file, "wb") as f:

        # Magic
        f.write(MAGIC)

        # Architecture
        f.write(
            struct.pack(
                "<IIII",
                INPUTS,
                HIDDEN1,
                HIDDEN2,
                OUTPUTS
            )
        )

        # W1
        f.write(
            struct.pack(
                "<" + "h" * len(w1_q),
                *w1_q
            )
        )

        # B1
        f.write(
            struct.pack(
                "<" + "h" * len(b1_q),
                *b1_q
            )
        )

        # W2
        f.write(
            struct.pack(
                "<" + "h" * len(w2_q),
                *w2_q
            )
        )

        # B2
        f.write(
            struct.pack(
                "<" + "h" * len(b2_q),
                *b2_q
            )
        )

        # WO
        f.write(
            struct.pack(
                "<" + "h" * len(wo_q),
                *wo_q
            )
        )

        # BO
        f.write(
            struct.pack(
                "<" + "h" * len(bo_q),
                *bo_q
            )
        )

    print()
    print("NNUE exported:")
    print(output_file)
    print(
        f"Size: {os.path.getsize(output_file):,} bytes"
    )


# ============================================================
# TRAIN
# ============================================================

def train(args):

    device = torch.device(
        "cpu"
        if args.cpu or not torch.cuda.is_available()
        else "cuda"
    )

    print()
    print("====================================")
    print(" SimpleLogics NNUE Training")
    print("====================================")
    print(
        f"Device: {device}"
    )
    print(
        f"Batch : {args.batch}"
    )
    print(
        f"Epochs: {args.epochs}"
    )
    print(
        f"Steps : {args.steps}"
    )
    print(
        f"LR    : {args.lr}"
    )
    print("====================================")
    print()

    dataset = PositionDataset(
        args.data,
        args.limit
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    model = SimpleLogicsNNUE()

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5
    )

    loss_function = nn.SmoothL1Loss()

    start_epoch = 0
    total_steps = 0

    # --------------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------------

    if args.resume:

        checkpoint = Path(
            args.checkpoint
        )

        if checkpoint.exists():

            print(
                f"Loading checkpoint: {checkpoint}"
            )

            data = torch.load(
                checkpoint,
                map_location=device
            )

            model.load_state_dict(
                data["model"]
            )

            if "optimizer" in data:
                optimizer.load_state_dict(
                    data["optimizer"]
                )

            start_epoch = data.get(
                "epoch",
                0
            )

            total_steps = data.get(
                "steps",
                0
            )

            print(
                f"Resumed epoch={start_epoch}, "
                f"steps={total_steps}"
            )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_loss = float("inf")

    for epoch in range(
        start_epoch,
        args.epochs
    ):

        model.train()

        epoch_loss = 0.0
        samples = 0

        # ----------------------------------------------------
        # IMPORTANT:
        # restart DataLoader every epoch.
        # Also respect --steps globally.
        # ----------------------------------------------------

        for x, y in loader:

            if args.steps > 0 and total_steps >= args.steps:
                break

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            prediction = model(x)

            loss = loss_function(
                prediction,
                y
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            loss_value = float(
                loss.detach().cpu()
            )

            epoch_loss += (
                loss_value * x.size(0)
            )

            samples += x.size(0)

            total_steps += 1

            if (
                args.save_every > 0
                and total_steps % args.save_every == 0
            ):

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "steps": total_steps,
                    },
                    args.checkpoint
                )

                print(
                    f"Checkpoint saved "
                    f"at step {total_steps}"
                )

            if total_steps % 100 == 0:

                print(
                    f"step={total_steps} "
                    f"loss={loss_value:.4f}"
                )

        if samples > 0:

            epoch_loss /= samples

        print()
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={epoch_loss:.6f} "
            f"steps={total_steps}"
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "steps": total_steps,
                    "loss": best_loss,
                },
                args.checkpoint
            )

            print(
                "Best checkpoint saved."
            )

        if (
            args.steps > 0
            and total_steps >= args.steps
        ):
            break

    # --------------------------------------------------------
    # Save final checkpoint
    # --------------------------------------------------------

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs,
            "steps": total_steps,
            "loss": best_loss,
        },
        args.checkpoint
    )

    # --------------------------------------------------------
    # Export NNUE
    # --------------------------------------------------------

    export_nnue(
        model,
        args.output
    )

    print()
    print("====================================")
    print(" TRAINING COMPLETE")
    print("====================================")
    print(
        f"Positions: {len(dataset):,}"
    )
    print(
        f"Steps    : {total_steps:,}"
    )
    print(
        f"Best loss: {best_loss:.6f}"
    )
    print(
        f"Output   : {args.output}"
    )
    print("====================================")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train SimpleLogics NNUE"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Dataset FEN | SCORE"
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
        action="store_true"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=256
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=5000
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
        default=1000
    )

    parser.add_argument(
        "--cpu",
        action="store_true"
    )

    args = parser.parse_args()

    if args.batch <= 0:
        raise ValueError(
            "--batch harus > 0"
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs harus > 0"
        )

    if args.steps < 0:
        raise ValueError(
            "--steps tidak boleh negatif"
        )

    train(args)


if __name__ == "__main__":
    main()
