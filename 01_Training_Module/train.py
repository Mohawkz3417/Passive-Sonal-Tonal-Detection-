"""
train.py
=========
Training loop for the tonal segmentation network.

Usage (from anywhere, paths are resolved relative to this file):
    python train.py --model segformer --epochs 20 --steps-per-epoch 100 \
                     --batch-size 8 --out ../11_Weights/segformer_b0_tonal.pt

Data is generated on-the-fly by 05_Dataset_Generator/dataset_builder.py, so
no dataset has to be downloaded to get started; real datasets (DeepShip /
ShipsEar / DS3500 / NPOL data if released) can be substituted later by
pointing a Dataset at pre-computed spectrogram/mask .npz pairs (see
materialize_dataset in dataset_builder.py) using the same tensor shapes.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import json

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "05_Dataset_Generator"))
sys.path.append(os.path.join(ROOT, "09_Models"))
sys.path.append(os.path.join(ROOT, "13_Scripts"))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_builder import TonalSegDataset      # noqa: E402
from segformer_tonal import segformer_b0_tonal    # noqa: E402
from unet_tonal import UNetTonal                  # noqa: E402
from losses import DiceBCELoss                    # noqa: E402
from config_loader import load_config, get as cfg_get  # noqa: E402

CFG = load_config()


def build_model(name: str):
    if name == "segformer":
        return segformer_b0_tonal()
    elif name == "unet":
        return UNetTonal()
    raise ValueError(f"unknown model {name}")


@torch.no_grad()
def pixel_metrics(logits, target, thresh=0.5):
    probs = torch.sigmoid(logits)
    pred = (probs > thresh).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[train] device = {device}")

    model = build_model(args.model).to(device)
    print(f"[train] model = {args.model}, params = {model.num_parameters()/1e6:.2f}M")

    start_epoch_offset = 0
    prior_history = []
    resumed_from = None
    if args.resume:
        resume_path = args.resume
        if os.path.isfile(resume_path):
            ckpt = torch.load(resume_path, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt)
            model.load_state_dict(state_dict)
            resumed_from = resume_path
            print(f"[train] resumed weights from {resume_path}")
            hist_path = os.path.splitext(resume_path)[0] + "_history.json"
            if os.path.isfile(hist_path):
                try:
                    with open(hist_path) as f:
                        prior_history = json.load(f)
                    start_epoch_offset = len(prior_history)
                    print(f"[train] loaded prior history ({start_epoch_offset} epochs already recorded)")
                except (json.JSONDecodeError, OSError):
                    prior_history = []
        else:
            print(f"[train] --resume path {resume_path} does not exist, starting from random init")

    train_ds = TonalSegDataset(length=args.steps_per_epoch * args.batch_size,
                                duration_s=args.duration, fs=args.fs,
                                n_fft=args.n_fft, hop_length=args.hop_length,
                                snr_range=(args.snr_lo, args.snr_hi),
                                hard_snr_range=(args.hard_snr_lo, args.hard_snr_hi),
                                hard_snr_prob=args.hard_snr_prob,
                                noise_profiles=args.noise_profiles.split(","),
                                f_min=args.f_min, f_max=args.f_max, doppler_prob=args.doppler_prob,
                                real_noise_dir=args.real_data_dir,
                                real_noise_prob=args.real_noise_prob)
    val_ds = TonalSegDataset(length=max(8, args.batch_size * 4), duration_s=args.duration,
                              fs=args.fs, n_fft=args.n_fft, hop_length=args.hop_length,
                              snr_range=(args.snr_lo, args.snr_hi),
                              hard_snr_range=(args.hard_snr_lo, args.hard_snr_hi),
                              hard_snr_prob=args.hard_snr_prob, seed=999,
                              noise_profiles=args.noise_profiles.split(","),
                              f_min=args.f_min, f_max=args.f_max, doppler_prob=args.doppler_prob,
                              real_noise_dir=args.real_data_dir,
                              real_noise_prob=args.real_noise_prob)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    criterion = DiceBCELoss(bce_weight=0.5, pos_weight=8.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    history = list(prior_history)
    best_f1 = max([h.get("f1", -1.0) for h in prior_history], default=-1.0)
    if resumed_from is not None and best_f1 >= 0:
        print(f"[train] best F1 so far (from prior history) = {best_f1:.3f}; "
              f"new checkpoint only overwrites if it improves on this")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss, parts = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        model.eval()
        val_metrics = {"precision": 0, "recall": 0, "f1": 0, "iou": 0}
        n_batches = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                m = pixel_metrics(logits, masks)
                for k in val_metrics:
                    val_metrics[k] += m[k]
                n_batches += 1
        for k in val_metrics:
            val_metrics[k] /= max(1, n_batches)

        dt = time.time() - t0
        train_loss = running_loss / len(train_loader)
        global_epoch = start_epoch_offset + epoch + 1
        print(f"[epoch {epoch+1:03d}/{args.epochs} | global {global_epoch:03d}] loss={train_loss:.4f} "
              f"val_P={val_metrics['precision']:.3f} val_R={val_metrics['recall']:.3f} "
              f"val_F1={val_metrics['f1']:.3f} val_IoU={val_metrics['iou']:.3f} "
              f"({dt:.1f}s)")
        history.append({"epoch": global_epoch, "train_loss": train_loss, **val_metrics, "time_s": dt,
                         "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size})

        # `args.out` is the RESUME file: with --save-last it always holds the most
        # recent epoch's weights (even a dip) so a fresh process can continue
        # training from where it left off instead of getting stuck replaying the
        # same epoch. That means args.out can legitimately be WORSE than the best
        # epoch ever seen across a multi-round resumed run. `*_best.pt` is a
        # separate sidecar that is ONLY ever overwritten when a new global best F1
        # is reached, so the actual best weights are never silently lost to a later
        # noisy/worse epoch -- this is what should be copied into 11_Weights/ for
        # deployment, not blindly whatever args.out happens to hold when training
        # stops.
        best_path = os.path.splitext(args.out)[0] + "_best.pt"
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model_name": args.model, "state_dict": model.state_dict(),
                        "config": vars(args), "global_epoch": global_epoch}, args.out)
            torch.save({"model_name": args.model, "state_dict": model.state_dict(),
                        "config": vars(args), "global_epoch": global_epoch,
                        "best_f1": best_f1}, best_path)
            print(f"  -> saved new best checkpoint (F1={best_f1:.3f}) to {args.out} "
                  f"(and sidecar {best_path})")
        elif args.save_last:
            torch.save({"model_name": args.model, "state_dict": model.state_dict(),
                        "config": vars(args), "global_epoch": global_epoch}, args.out)
            print(f"  -> F1 did not improve ({val_metrics['f1']:.3f} <= {best_f1:.3f}); "
                  f"saved LAST checkpoint anyway (--save-last) to {args.out} -- "
                  f"true best (F1={best_f1:.3f}) remains untouched at {best_path}")

    with open(os.path.splitext(args.out)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("[train] done.")
    return history


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["segformer", "unet"],
                    default=cfg_get(CFG, "training.model", "segformer"))
    p.add_argument("--epochs", type=int, default=cfg_get(CFG, "training.epochs", 20))
    p.add_argument("--steps-per-epoch", type=int, default=cfg_get(CFG, "training.steps_per_epoch", 100))
    p.add_argument("--batch-size", type=int, default=cfg_get(CFG, "training.batch_size", 8))
    p.add_argument("--lr", type=float, default=cfg_get(CFG, "training.lr", 3e-4))
    p.add_argument("--duration", type=float, default=cfg_get(CFG, "training.duration_s", 4.0))
    p.add_argument("--fs", type=float, default=cfg_get(CFG, "signal.sampling_rate_hz", 16000.0))
    p.add_argument("--n-fft", type=int, default=cfg_get(CFG, "signal.fft_size", 1024))
    p.add_argument("--hop-length", type=int, default=cfg_get(CFG, "signal.hop_length", 256))
    p.add_argument("--num-workers", type=int, default=cfg_get(CFG, "training.num_workers", 0))
    p.add_argument("--snr-lo", type=float, default=cfg_get(CFG, "training.snr_lo_db", -10.0))
    p.add_argument("--snr-hi", type=float, default=cfg_get(CFG, "training.snr_hi_db", 20.0))
    p.add_argument("--hard-snr-prob", type=float, default=cfg_get(CFG, "training.hard_snr_prob", 0.5),
                    help="fraction of training/val samples drawn from the hard-SNR band")
    p.add_argument("--hard-snr-lo", type=float, default=cfg_get(CFG, "training.hard_snr_range_db", [-10.0, -7.0])[0])
    p.add_argument("--hard-snr-hi", type=float, default=cfg_get(CFG, "training.hard_snr_range_db", [-10.0, -7.0])[1])
    p.add_argument("--noise-profiles", type=str,
                    default=",".join(cfg_get(CFG, "training.noise_profiles",
                                              ["shipping", "environmental", "biological", "background"])),
                    help="comma-separated list, one chosen at random per training sample")
    p.add_argument("--f-min", type=float, default=cfg_get(CFG, "training.f_min_hz", 150.0),
                    help="minimum tonal frequency actually used by the training generator")
    p.add_argument("--f-max", type=float, default=cfg_get(CFG, "training.f_max_hz", 3500.0),
                    help="maximum tonal frequency actually used by the training generator")
    p.add_argument("--doppler-prob", type=float, default=cfg_get(CFG, "training.doppler_prob", 0.3),
                    help="fraction of tonals that get a Doppler frequency ramp")
    p.add_argument("--real-data-dir", type=str, default=cfg_get(CFG, "training.real_data_dir", None),
                    help="folder containing ShipsEar/DeepShip/DS3500/etc. WAV or FLAC files, "
                         "used as a real-noise background pool (see REAL_DATA_GUIDE.md)")
    p.add_argument("--real-noise-prob", type=float, default=cfg_get(CFG, "training.real_noise_prob", 0.5),
                    help="fraction of samples that use a real recording as the noise "
                         "background instead of synthetic noise (only used if --real-data-dir is set)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", type=str,
                    default=os.path.join(ROOT, cfg_get(CFG, "training.output_checkpoint",
                                                          "11_Weights/active/segformer_b0_tonal.pt")))
    p.add_argument("--resume", type=str, default=None,
                    help="path to an existing checkpoint whose weights (and, if present, "
                         "matching *_history.json) are loaded before training continues. "
                         "Lets a long training run be split across several bounded "
                         "invocations without losing progress -- this does not change "
                         "the model architecture or training algorithm, only where the "
                         "weights start from.")
    p.add_argument("--save-last", action="store_true",
                    help="also save the checkpoint at the end of an epoch where val F1 did "
                         "not improve (in addition to the normal best-F1 save), so a resumed "
                         "run always has fresh weights to continue from even during a plateau")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
