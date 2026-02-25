#!/usr/bin/env python3
# =========================
# FILE: train_regression.py
# =========================

import os
import re
import json
import argparse
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from scripts.simpleForms_Regression.synth_dataset_regression import SynthMultiLabelNPZRegression
from models.simpleForms_Regression.tsm_cnn_regression import TSM_CNN_Regression


# -------------------------------------------------------
# Case-Namen (Ordnerstruktur / Auswertung konsistent halten)
# -------------------------------------------------------
CASE_ORDER = ["FIX_MOT_FIX_ROT", "FIX_MOT_VAR_ROT", "VAR_MOT_FIX_ROT", "VAR_MOT_VAR_ROT"]


@dataclass
class Metrics:
    # Aggregierte Metriken pro Epoch (Train/Val/Test)
    loss: float
    acc_motion: float
    acc_rot: float
    acc_shape: float
    acc_exact: float
    mae_speed: float
    mae_omega: float
    rmse_speed: float
    rmse_omega: float


def set_seed(seed: int) -> None:
    # Reproduzierbarkeit (ohne Determinismus-Zwang)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def infer_case_from_dataset_path(dataset_root: str) -> str:
    # Erwartet Ordnernamen wie <CASE>_s123 -> gibt <CASE>
    name = os.path.basename(os.path.normpath(dataset_root))
    m = re.match(r"^(.*)_s\d+$", name)
    return m.group(1) if m else name


def make_run_dir(base_save_dir: str, case: str, use_tsm: bool, seed: int) -> str:
    """
    Baut die gewünschte Ordnerstruktur:
      base_save_dir/
        with_TSM_res/<CASE>/<TIMESTAMP>_s123_tsm/
        without_TSM_res/<CASE>/<TIMESTAMP>_s123_noTSM/
    """
    variant = "with_TSM_res" if use_tsm else "without_TSM_res"
    run_name = time.strftime("%Y%m%d-%H%M%S") + f"_s{seed}" + ("_tsm" if use_tsm else "_noTSM")
    out_dir = os.path.join(base_save_dir, variant, case, run_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def freeze_batchnorm(model: nn.Module) -> None:
    # BatchNorm in eval setzen (running stats nicht updaten, keine batch-stats)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def _make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    # DataLoader kwargs zentral bündeln
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
        )
    return DataLoader(ds, **kwargs)


def save_curve_plot(
    history: pd.DataFrame,
    xcol: str,
    ycols: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_png: str,
) -> None:
    plt.figure()
    for yc in ycols:
        plt.plot(history[xcol].values, history[yc].values, label=yc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_scatter(true: np.ndarray, pred: np.ndarray, title: str, xlabel: str, ylabel: str, out_png: str) -> None:
    plt.figure()
    plt.scatter(true, pred, s=8, alpha=0.5)
    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_residuals(true: np.ndarray, pred: np.ndarray, title: str, xlabel: str, ylabel: str, out_png: str) -> None:
    plt.figure()
    resid = pred - true
    plt.scatter(true, resid, s=8, alpha=0.5)
    plt.axhline(0.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str,
    out_png: str,
    normalize: bool = True,
) -> None:
    cm = cm.astype(np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1.0
        cm = cm / row_sum

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # Werte einzeichnen
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                     ha="center", va="center")

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=220)
    plt.close()


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def save_binned_metrics(
    true_speed: np.ndarray,
    pred_speed: np.ndarray,
    true_omega: np.ndarray,
    pred_omega: np.ndarray,
    acc_exact_mask: np.ndarray,
    title_prefix: str,
    out_dir: str,
    num_bins: int = 6,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _bin_edges(x: np.ndarray, bins: int) -> np.ndarray:
        x = x.astype(np.float64)
        lo = float(np.min(x))
        hi = float(np.max(x))
        if hi <= lo:
            hi = lo + 1e-6
        return np.linspace(lo, hi, bins + 1)

    def _binned(x: np.ndarray, mae_target: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        edges = _bin_edges(x, bins)
        idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
        mae = np.full((bins,), np.nan, dtype=np.float64)
        acc = np.full((bins,), np.nan, dtype=np.float64)
        cnt = np.zeros((bins,), dtype=np.int64)
        for b in range(bins):
            m = (idx == b)
            cnt[b] = int(m.sum())
            if cnt[b] > 0:
                mae[b] = float(np.mean(np.abs(mae_target[m])))
                acc[b] = float(np.mean(acc_exact_mask[m].astype(np.float64)))
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, mae, acc

    # Speed bins
    centers_s, mae_s, acc_s = _binned(true_speed, pred_speed - true_speed, num_bins)
    plt.figure()
    plt.plot(centers_s, mae_s, marker="o")
    plt.xlabel("True speed (binned)")
    plt.ylabel("MAE(speed)")
    plt.title(f"{title_prefix} | MAE(speed) vs true speed bins")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "binned_mae_speed.png"), dpi=220)
    plt.close()

    plt.figure()
    plt.plot(centers_s, acc_s, marker="o")
    plt.xlabel("True speed (binned)")
    plt.ylabel("Acc exact")
    plt.title(f"{title_prefix} | Acc(exact) vs true speed bins")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "binned_acc_exact_by_speed.png"), dpi=220)
    plt.close()

    # Omega bins
    centers_w, mae_w, acc_w = _binned(true_omega, pred_omega - true_omega, num_bins)
    plt.figure()
    plt.plot(centers_w, mae_w, marker="o")
    plt.xlabel("True omega (binned)")
    plt.ylabel("MAE(omega)")
    plt.title(f"{title_prefix} | MAE(omega) vs true omega bins")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "binned_mae_omega.png"), dpi=220)
    plt.close()

    plt.figure()
    plt.plot(centers_w, acc_w, marker="o")
    plt.xlabel("True omega (binned)")
    plt.ylabel("Acc exact")
    plt.title(f"{title_prefix} | Acc(exact) vs true omega bins")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "binned_acc_exact_by_omega.png"), dpi=220)
    plt.close()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ce_motion: nn.Module,
    ce_rot: nn.Module,
    ce_shape: nn.Module,
    reg_loss_fn: nn.Module,
    lambda_speed: float,
    lambda_omega: float,
    use_amp: bool = False,
) -> Metrics:
    model.eval()

    total_loss = 0.0
    total = 0

    correct_m = 0
    correct_r = 0
    correct_s = 0
    correct_exact = 0

    abs_err_speed_sum = 0.0
    abs_err_omega_sum = 0.0
    sq_err_speed_sum = 0.0
    sq_err_omega_sum = 0.0

    # CE-Komponenten separat tracken
    sum_loss_m = 0.0
    sum_loss_r = 0.0
    sum_loss_s = 0.0

    amp_enabled = (device.type == "cuda" and bool(use_amp))

    # Worst-batch Diagnose
    worst = {
        "loss": -1.0,
        "loss_m": None,
        "loss_r": None,
        "loss_s": None,
        "loss_cls": None,
        "ls": None,
        "lo": None,
        "acc_m": None,
        "acc_r": None,
        "acc_s": None,
        "acc_exact": None,
    }

    for clips, motion, rot, shape, speed, omega in loader:
        clips = clips.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        rot = rot.to(device, non_blocking=True)
        shape = shape.to(device, non_blocking=True)
        speed = speed.to(device, non_blocking=True).float()
        omega = omega.to(device, non_blocking=True).float()

        with autocast(device_type=device.type, enabled=amp_enabled):
            logits_m, logits_r, logits_s, pred_speed, pred_omega = model(clips)

            pred_speed = pred_speed.view(-1)
            pred_omega = pred_omega.view(-1)
            speed = speed.view(-1)
            omega = omega.view(-1)

            loss_m = ce_motion(logits_m, motion)
            loss_r = ce_rot(logits_r, rot)
            loss_s = ce_shape(logits_s, shape)
            loss_cls = loss_m + loss_r + loss_s

            loss_reg_speed = reg_loss_fn(pred_speed, speed)
            loss_reg_omega = reg_loss_fn(pred_omega, omega)

            loss = loss_cls + (lambda_speed * loss_reg_speed) + (lambda_omega * loss_reg_omega)

            loss_v = float(loss.item())
            if not np.isfinite(loss_v):
                print("[VAL] non-finite loss!")
                break

        pred_m = logits_m.argmax(dim=1)
        pred_r = logits_r.argmax(dim=1)
        pred_s = logits_s.argmax(dim=1)

        bsz = clips.size(0)

        bm = float((pred_m == motion).float().mean().item())
        br = float((pred_r == rot).float().mean().item())
        bs = float((pred_s == shape).float().mean().item())
        be = float(((pred_m == motion) & (pred_r == rot) & (pred_s == shape)).float().mean().item())

        if loss_v > worst["loss"]:
            worst["loss"] = loss_v
            worst["loss_m"] = float(loss_m.item())
            worst["loss_r"] = float(loss_r.item())
            worst["loss_s"] = float(loss_s.item())
            worst["loss_cls"] = float(loss_cls.item())
            worst["ls"] = float(loss_reg_speed.item())
            worst["lo"] = float(loss_reg_omega.item())
            worst["acc_m"] = bm
            worst["acc_r"] = br
            worst["acc_s"] = bs
            worst["acc_exact"] = be

        total += bsz
        total_loss += float(loss.item()) * bsz

        correct_m += int((pred_m == motion).sum().item())
        correct_r += int((pred_r == rot).sum().item())
        correct_s += int((pred_s == shape).sum().item())
        correct_exact += int(((pred_m == motion) & (pred_r == rot) & (pred_s == shape)).sum().item())

        sum_loss_m += float(loss_m.item()) * bsz
        sum_loss_r += float(loss_r.item()) * bsz
        sum_loss_s += float(loss_s.item()) * bsz

        err_speed = (pred_speed - speed)
        err_omega = (pred_omega - omega)

        abs_err_speed_sum += float(err_speed.abs().sum().item())
        abs_err_omega_sum += float(err_omega.abs().sum().item())

        sq_err_speed_sum += float((err_speed ** 2).sum().item())
        sq_err_omega_sum += float((err_omega ** 2).sum().item())

    denom = max(total, 1)

    mae_speed = abs_err_speed_sum / denom
    mae_omega = abs_err_omega_sum / denom
    rmse_speed = float(np.sqrt(sq_err_speed_sum / denom))
    rmse_omega = float(np.sqrt(sq_err_omega_sum / denom))

    if worst["loss_cls"] is not None:
        print(
            f"[VAL] worst batch loss={worst['loss']:.3f} | "
            f"m={worst['loss_m']:.3f} r={worst['loss_r']:.3f} s={worst['loss_s']:.3f} | "
            f"cls={worst['loss_cls']:.3f} | reg_s={worst['ls']:.3f} | reg_w={worst['lo']:.3f} || "
            f"acc(m/r/s/exact)={worst['acc_m']:.2f}/{worst['acc_r']:.2f}/{worst['acc_s']:.2f}/{worst['acc_exact']:.2f}"
        )
        print(
            f"[VAL] mean CE: m={sum_loss_m/denom:.4f} r={sum_loss_r/denom:.4f} s={sum_loss_s/denom:.4f}"
        )

    return Metrics(
        loss=total_loss / denom,
        acc_motion=correct_m / denom,
        acc_rot=correct_r / denom,
        acc_shape=correct_s / denom,
        acc_exact=correct_exact / denom,
        mae_speed=mae_speed,
        mae_omega=mae_omega,
        rmse_speed=rmse_speed,
        rmse_omega=rmse_omega,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    ce_motion: nn.Module,
    ce_rot: nn.Module,
    ce_shape: nn.Module,
    reg_loss_fn: nn.Module,
    lambda_speed: float,
    lambda_omega: float,
    use_amp: bool,
    scaler: GradScaler,
    freeze_bn: bool,
) -> Metrics:
    model.train()
    if freeze_bn:
        freeze_batchnorm(model)

    total_loss = 0.0
    total = 0

    correct_m = 0
    correct_r = 0
    correct_s = 0
    correct_exact = 0

    abs_err_speed_sum = 0.0
    abs_err_omega_sum = 0.0
    sq_err_speed_sum = 0.0
    sq_err_omega_sum = 0.0

    amp_enabled = (device.type == "cuda" and bool(use_amp))

    for clips, motion, rot, shape, speed, omega in loader:
        clips = clips.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        rot = rot.to(device, non_blocking=True)
        shape = shape.to(device, non_blocking=True)
        speed = speed.to(device, non_blocking=True).float()
        omega = omega.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=amp_enabled):
            logits_m, logits_r, logits_s, pred_speed, pred_omega = model(clips)

            pred_speed = pred_speed.view(-1)
            pred_omega = pred_omega.view(-1)
            speed = speed.view(-1)
            omega = omega.view(-1)

            loss_cls = (
                ce_motion(logits_m, motion)
                + ce_rot(logits_r, rot)
                + ce_shape(logits_s, shape)
            )
            loss_reg_speed = reg_loss_fn(pred_speed, speed)
            loss_reg_omega = reg_loss_fn(pred_omega, omega)
            loss = loss_cls + (lambda_speed * loss_reg_speed) + (lambda_omega * loss_reg_omega)

        loss_v = float(loss.item())
        if not np.isfinite(loss_v):
            raise RuntimeError("non-finite training loss detected")

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        pred_m = logits_m.argmax(dim=1)
        pred_r = logits_r.argmax(dim=1)
        pred_s = logits_s.argmax(dim=1)

        bsz = clips.size(0)
        total += bsz
        total_loss += float(loss.item()) * bsz

        correct_m += int((pred_m == motion).sum().item())
        correct_r += int((pred_r == rot).sum().item())
        correct_s += int((pred_s == shape).sum().item())
        correct_exact += int(((pred_m == motion) & (pred_r == rot) & (pred_s == shape)).sum().item())

        err_speed = (pred_speed - speed)
        err_omega = (pred_omega - omega)

        abs_err_speed_sum += float(err_speed.abs().sum().item())
        abs_err_omega_sum += float(err_omega.abs().sum().item())
        sq_err_speed_sum += float((err_speed ** 2).sum().item())
        sq_err_omega_sum += float((err_omega ** 2).sum().item())

    denom = max(total, 1)
    mae_speed = abs_err_speed_sum / denom
    mae_omega = abs_err_omega_sum / denom
    rmse_speed = float(np.sqrt(sq_err_speed_sum / denom))
    rmse_omega = float(np.sqrt(sq_err_omega_sum / denom))

    return Metrics(
        loss=total_loss / denom,
        acc_motion=correct_m / denom,
        acc_rot=correct_r / denom,
        acc_shape=correct_s / denom,
        acc_exact=correct_exact / denom,
        mae_speed=mae_speed,
        mae_omega=mae_omega,
        rmse_speed=rmse_speed,
        rmse_omega=rmse_omega,
    )


def make_selection_score(m: Metrics, mode: str, alpha: float) -> float:
    # Höher = besser
    if mode == "exact":
        return float(m.acc_exact)
    if mode == "mae_sum":
        return -float(m.mae_speed + m.mae_omega)
    if mode == "score":
        return float(m.acc_exact) - alpha * float(m.mae_speed + m.mae_omega)
    raise ValueError(f"Unknown select_metric: {mode}")


@torch.no_grad()
def collect_test_preds_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, np.ndarray]:
    # Sammelt True/Pred für alle Heads + Regression
    model.eval()

    amp_enabled = (device.type == "cuda" and bool(use_amp))

    true_m, true_r, true_s = [], [], []
    pred_m, pred_r, pred_s = [], [], []
    true_speed, pred_speed = [], []
    true_omega, pred_omega = [], []

    for clips, motion, rot, shape, speed, omega in loader:
        clips = clips.to(device, non_blocking=True)
        motion = motion.to(device, non_blocking=True)
        rot = rot.to(device, non_blocking=True)
        shape = shape.to(device, non_blocking=True)
        speed = speed.to(device, non_blocking=True).float()
        omega = omega.to(device, non_blocking=True).float()

        with autocast(device_type=device.type, enabled=amp_enabled):
            logits_m, logits_r, logits_s, ps, po = model(clips)

        pm = logits_m.argmax(dim=1)
        pr = logits_r.argmax(dim=1)
        ps_cls = logits_s.argmax(dim=1)

        true_m.append(motion.detach().cpu().numpy())
        true_r.append(rot.detach().cpu().numpy())
        true_s.append(shape.detach().cpu().numpy())
        pred_m.append(pm.detach().cpu().numpy())
        pred_r.append(pr.detach().cpu().numpy())
        pred_s.append(ps_cls.detach().cpu().numpy())

        true_speed.append(speed.detach().cpu().numpy())
        pred_speed.append(ps.detach().cpu().numpy())
        true_omega.append(omega.detach().cpu().numpy())
        pred_omega.append(po.detach().cpu().numpy())

    return {
        "true_motion": np.concatenate(true_m, axis=0),
        "true_rot": np.concatenate(true_r, axis=0),
        "true_shape": np.concatenate(true_s, axis=0),
        "pred_motion": np.concatenate(pred_m, axis=0),
        "pred_rot": np.concatenate(pred_r, axis=0),
        "pred_shape": np.concatenate(pred_s, axis=0),
        "true_speed": np.concatenate(true_speed, axis=0).reshape(-1),
        "pred_speed": np.concatenate(pred_speed, axis=0).reshape(-1),
        "true_omega": np.concatenate(true_omega, axis=0).reshape(-1),
        "pred_omega": np.concatenate(pred_omega, axis=0).reshape(-1),
    }


@torch.no_grad()
def bench_inference_ms(
    model: nn.Module,
    device: torch.device,
    batch: torch.Tensor,
    iters: int = 200,
    warmup: int = 50,
    use_amp: bool = False,
) -> float:
    # Einfache Wall-Clock Messung (ms pro forward)
    model.eval()
    amp_enabled = (device.type == "cuda" and bool(use_amp))

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Warmup
    for _ in range(max(warmup, 0)):
        with autocast(device_type=device.type, enabled=amp_enabled):
            _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(max(iters, 1)):
        with autocast(device_type=device.type, enabled=amp_enabled):
            _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    return float((t1 - t0) * 1000.0 / max(iters, 1))


def main() -> None:
    torch.backends.cudnn.benchmark = True

    ap = argparse.ArgumentParser()

    # Dataset
    ap.add_argument("--data", type=str, required=True, help="Pfad zum Dataset-Root")
    ap.add_argument("--device", type=str, default="cuda", help="cuda oder cpu")
    ap.add_argument("--num_workers", type=int, default=2)

    # Model / TSM
    ap.add_argument("--use_tsm", action="store_true", help="TSM aktivieren")
    ap.add_argument("--n_segment", type=int, default=16)
    ap.add_argument("--fold_div", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)

    # Training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=123)

    # Early Stopping
    ap.add_argument("--early_stop", action="store_true", help="Early stopping aktivieren")
    ap.add_argument("--patience", type=int, default=15, help="Epochen ohne Verbesserung bis Abbruch")
    ap.add_argument("--min_delta", type=float, default=0.001, help="Mindestverbesserung, um als 'besser' zu zählen")

    # Regression
    ap.add_argument(
        "--normalize_regression",
        action="store_true",
        help="speed/omega als [0,1] normalisiert laden (aus dataset_config.json).",
    )
    ap.add_argument("--lambda_speed", type=float, default=1.0)
    ap.add_argument("--lambda_omega", type=float, default=1.0)
    ap.add_argument("--reg_loss", type=str, default="smoothl1", choices=["smoothl1", "mse"])

    # Best checkpoint Auswahl
    ap.add_argument("--select_metric", type=str, default="score", choices=["score", "exact", "mae_sum"])
    ap.add_argument("--alpha", type=float, default=0.5)

    # Output base dir (wird beim Aufruf gesetzt)
    ap.add_argument("--save_dir", type=str, default="runs/ResShift/Regression")

    # AMP
    ap.add_argument("--amp", action="store_true", help="Automatic Mixed Precision")

    # BatchNorm Freeze
    ap.add_argument("--freeze_bn", action="store_true", help="BatchNorm2d im Training einfrieren")

    # Optional inference benchmark (für spätere Tradeoff-Plots)
    ap.add_argument("--bench_infer", action="store_true", help="Misst Inferenzzeit (ms/forward) auf 1 Batch")
    ap.add_argument("--bench_iters", type=int, default=200)
    ap.add_argument("--bench_warmup", type=int, default=50)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    case = infer_case_from_dataset_path(args.data)

    out_dir = make_run_dir(args.save_dir, case=case, use_tsm=args.use_tsm, seed=args.seed)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Datasets
    ds_train = SynthMultiLabelNPZRegression(args.data, "train", normalize_regression=args.normalize_regression)
    ds_val = SynthMultiLabelNPZRegression(args.data, "val", normalize_regression=args.normalize_regression)
    ds_test = SynthMultiLabelNPZRegression(args.data, "test", normalize_regression=args.normalize_regression)

    # DataLoader
    dl_train = _make_loader(ds_train, args.batch_size, True, args.num_workers, device)
    dl_val = _make_loader(ds_val, args.batch_size, False, args.num_workers, device)
    dl_test = _make_loader(ds_test, args.batch_size, False, args.num_workers, device)

    # Modell
    model = TSM_CNN_Regression(
        n_segment=args.n_segment,
        fold_div=args.fold_div,
        use_tsm=args.use_tsm,
        dropout_p=args.dropout,
        normalize_regression=args.normalize_regression,
    ).to(device)

    # Losses
    ce_motion = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_rot = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_shape = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.reg_loss == "smoothl1":
        reg_loss_fn = nn.SmoothL1Loss(beta=1.0)
    else:
        reg_loss_fn = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    amp_enabled = (device.type == "cuda" and bool(args.amp))
    scaler = GradScaler("cuda", enabled=amp_enabled)

    # History
    history_rows: List[Dict[str, Any]] = []

    # Best checkpoint
    best_path = os.path.join(out_dir, "best.pt")
    best_score = -1e18
    best_epoch = 0
    epochs_no_improve = 0

    # Args dump
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Device: {device}")
    print(f"Case: {case}")
    print(f"Dataset: {args.data}")
    print(f"Train/Val/Test: {len(ds_train)} / {len(ds_val)} / {len(ds_test)}")
    print(f"Normalize regression: {args.normalize_regression}")
    print(f"AMP: {args.amp}")
    print(f"freeze_bn: {args.freeze_bn}")
    print(f"Output: {out_dir}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, dl_train, device, optimizer,
            ce_motion, ce_rot, ce_shape,
            reg_loss_fn,
            lambda_speed=args.lambda_speed,
            lambda_omega=args.lambda_omega,
            use_amp=bool(args.amp),
            scaler=scaler,
            freeze_bn=bool(args.freeze_bn),
        )

        val_m = evaluate(
            model,
            dl_val,
            device,
            ce_motion,
            ce_rot,
            ce_shape,
            reg_loss_fn,
            lambda_speed=args.lambda_speed,
            lambda_omega=args.lambda_omega,
            use_amp=False,
        )

        dt = time.time() - t0
        score = make_selection_score(val_m, mode=args.select_metric, alpha=args.alpha)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | {dt:6.1f}s | "
            f"train loss {train_m.loss:.4f} exact {train_m.acc_exact:.3f} "
            f"mae_s {train_m.mae_speed:.3f} mae_w {train_m.mae_omega:.3f} || "
            f"val loss {val_m.loss:.4f} exact {val_m.acc_exact:.3f} "
            f"mae_s {val_m.mae_speed:.3f} mae_w {val_m.mae_omega:.3f} | score {score:.4f}"
        )

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_m.loss,
                "val_loss": val_m.loss,
                "train_acc_exact": train_m.acc_exact,
                "val_acc_exact": val_m.acc_exact,
                "train_mae_speed": train_m.mae_speed,
                "val_mae_speed": val_m.mae_speed,
                "train_mae_omega": train_m.mae_omega,
                "val_mae_omega": val_m.mae_omega,
                "train_rmse_speed": train_m.rmse_speed,
                "val_rmse_speed": val_m.rmse_speed,
                "train_rmse_omega": train_m.rmse_omega,
                "val_rmse_omega": val_m.rmse_omega,
                "score": score,
            }
        )

        improved = score > (best_score + args.min_delta)
        if improved:
            best_score = score
            best_epoch = epoch
            epochs_no_improve = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_m.__dict__,
                    "args": vars(args),
                    "best_score": float(best_score),
                    "best_epoch": int(best_epoch),
                    "freeze_bn": bool(args.freeze_bn),
                    "speed_max": float(getattr(ds_train, "speed_max", np.nan)) if hasattr(ds_train, "speed_max") else float("nan"),
                    "omega_max": float(getattr(ds_train, "omega_max", np.nan)) if hasattr(ds_train, "omega_max") else float("nan"),
                },
                best_path,
            )
        else:
            epochs_no_improve += 1

        if args.early_stop and epochs_no_improve >= args.patience:
            print(
                f"\n[EARLY STOP] Keine Verbesserung > min_delta={args.min_delta} "
                f"für {args.patience} Epochen. Stoppe bei Epoche {epoch}. "
                f"Best epoch: {best_epoch} | Best score: {best_score:.4f}"
            )
            break

        scheduler.step()

    hist_df = pd.DataFrame(history_rows)
    hist_csv = os.path.join(out_dir, "history.csv")
    hist_df.to_csv(hist_csv, index=False)

    print("\nTraining finished.")
    print(f"Best selection score: {best_score:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best checkpoint: {best_path}")
    print(f"History: {hist_csv}")

    # --- TEST: best checkpoint laden ---
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_m = evaluate(
        model,
        dl_test,
        device,
        ce_motion,
        ce_rot,
        ce_shape,
        reg_loss_fn,
        lambda_speed=args.lambda_speed,
        lambda_omega=args.lambda_omega,
        use_amp=False,
    )

    speed_max = float(ckpt.get("speed_max", float("nan")))
    omega_max = float(ckpt.get("omega_max", float("nan")))

    # Test-Preds (für Plots + spätere BA-Auswertung)
    preds = collect_test_preds_full(model, dl_test, device=device, use_amp=bool(args.amp))

    # Denorm für Plot/Analyse (falls Targets normalisiert)
    ts = preds["true_speed"].copy()
    ps = preds["pred_speed"].copy()
    tw = preds["true_omega"].copy()
    pw = preds["pred_omega"].copy()

    if args.normalize_regression and np.isfinite(speed_max) and speed_max > 0:
        ts_plot = ts * speed_max
        ps_plot = ps * speed_max
        speed_label = "Speed (px/frame)"
    else:
        ts_plot = ts
        ps_plot = ps
        speed_label = "Speed (normalized)" if args.normalize_regression else "Speed"

    if args.normalize_regression and np.isfinite(omega_max) and omega_max > 0:
        tw_plot = tw * omega_max
        pw_plot = pw * omega_max
        omega_label = "Omega (deg/frame)"
    else:
        tw_plot = tw
        pw_plot = pw
        omega_label = "Omega (normalized)" if args.normalize_regression else "Omega"

    # Inference benchmark (optional)
    infer_ms: Optional[float] = None
    if args.bench_infer:
        first_batch = next(iter(dl_test))[0].to(device, non_blocking=True)
        infer_ms = bench_inference_ms(
            model=model,
            device=device,
            batch=first_batch,
            iters=int(args.bench_iters),
            warmup=int(args.bench_warmup),
            use_amp=bool(args.amp),
        )

    denorm: Dict[str, Any] = {}
    if args.normalize_regression and np.isfinite(speed_max) and np.isfinite(omega_max) and speed_max > 0 and omega_max > 0:
        denorm = {
            "test_mae_speed_denorm": float(test_m.mae_speed * speed_max),
            "test_rmse_speed_denorm": float(test_m.rmse_speed * speed_max),
            "test_mae_omega_denorm": float(test_m.mae_omega * omega_max),
            "test_rmse_omega_denorm": float(test_m.rmse_omega * omega_max),
            "speed_max": speed_max,
            "omega_max": omega_max,
        }

    results = {
        "case": case,
        "seed": int(args.seed),
        "use_tsm": bool(args.use_tsm),
        "n_segment": int(args.n_segment),
        "fold_div": int(args.fold_div),
        "epochs_max": int(args.epochs),
        "best_epoch": int(best_epoch),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "normalize_regression": bool(args.normalize_regression),
        "lambda_speed": float(args.lambda_speed),
        "lambda_omega": float(args.lambda_omega),
        "select_metric": str(args.select_metric),
        "alpha": float(args.alpha),
        "early_stop": bool(args.early_stop),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "amp": bool(args.amp),
        "freeze_bn": bool(args.freeze_bn),

        "test_loss": float(test_m.loss),
        "test_acc_motion": float(test_m.acc_motion),
        "test_acc_rot": float(test_m.acc_rot),
        "test_acc_shape": float(test_m.acc_shape),
        "test_acc_exact": float(test_m.acc_exact),

        "test_mae_speed": float(test_m.mae_speed),
        "test_rmse_speed": float(test_m.rmse_speed),
        "test_mae_omega": float(test_m.mae_omega),
        "test_rmse_omega": float(test_m.rmse_omega),

        **denorm,
    }

    if infer_ms is not None:
        results["inference_ms_per_forward"] = float(infer_ms)
        print(f"[BENCH] inference_ms_per_forward={infer_ms:.3f} ms")

    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Persistiere per-sample predictions (für spätere cross-run/BA-Auswertung)
    pred_csv = os.path.join(out_dir, "test_preds.csv")
    pd.DataFrame(
        {
            "true_motion": preds["true_motion"],
            "pred_motion": preds["pred_motion"],
            "true_rot": preds["true_rot"],
            "pred_rot": preds["pred_rot"],
            "true_shape": preds["true_shape"],
            "pred_shape": preds["pred_shape"],
            "true_speed": ts_plot,
            "pred_speed": ps_plot,
            "true_omega": tw_plot,
            "pred_omega": pw_plot,
        }
    ).to_csv(pred_csv, index=False)

    print(
        f"\nTEST | loss {test_m.loss:.4f} | exact {test_m.acc_exact:.3f} | "
        f"mae_s {test_m.mae_speed:.4f} mae_w {test_m.mae_omega:.4f}"
    )

    # -------------------------
    # PLOTS (Kurven)
    # -------------------------
    save_curve_plot(
        hist_df, "epoch", ["train_loss", "val_loss"],
        "Loss over epochs", "Epoch", "Loss",
        os.path.join(plots_dir, "curves_loss.png"),
    )
    save_curve_plot(
        hist_df, "epoch", ["train_acc_exact", "val_acc_exact"],
        "Exact-match accuracy over epochs", "Epoch", "Accuracy",
        os.path.join(plots_dir, "curves_acc_exact.png"),
    )
    save_curve_plot(
        hist_df, "epoch", ["train_mae_speed", "val_mae_speed"],
        "MAE(speed) over epochs", "Epoch", "MAE",
        os.path.join(plots_dir, "curves_mae_speed.png"),
    )
    save_curve_plot(
        hist_df, "epoch", ["train_mae_omega", "val_mae_omega"],
        "MAE(omega) over epochs", "Epoch", "MAE",
        os.path.join(plots_dir, "curves_mae_omega.png"),
    )

    # -------------------------
    # PLOTS (Scatter/Residuals)
    # -------------------------
    save_scatter(
        ts_plot, ps_plot,
        "True vs Pred Speed (test)",
        f"True {speed_label}", f"Pred {speed_label}",
        os.path.join(plots_dir, "scatter_speed.png"),
    )
    save_scatter(
        tw_plot, pw_plot,
        "True vs Pred Omega (test)",
        f"True {omega_label}", f"Pred {omega_label}",
        os.path.join(plots_dir, "scatter_omega.png"),
    )
    save_residuals(
        ts_plot, ps_plot,
        "Residuals Speed (Pred-True) vs True (test)",
        f"True {speed_label}", "Residual",
        os.path.join(plots_dir, "residuals_speed.png"),
    )
    save_residuals(
        tw_plot, pw_plot,
        "Residuals Omega (Pred-True) vs True (test)",
        f"True {omega_label}", "Residual",
        os.path.join(plots_dir, "residuals_omega.png"),
    )

    # -------------------------
    # PLOTS (Confusion Matrices)
    # -------------------------
    cm_m = _confusion_matrix(preds["true_motion"], preds["pred_motion"], num_classes=5)
    cm_r = _confusion_matrix(preds["true_rot"], preds["pred_rot"], num_classes=3)
    cm_s = _confusion_matrix(preds["true_shape"], preds["pred_shape"], num_classes=2)

    save_confusion_matrix(cm_m, [str(i) for i in range(5)], "Confusion motion (test)", os.path.join(plots_dir, "cm_motion.png"))
    save_confusion_matrix(cm_r, [str(i) for i in range(3)], "Confusion rot (test)", os.path.join(plots_dir, "cm_rot.png"))
    save_confusion_matrix(cm_s, [str(i) for i in range(2)], "Confusion shape (test)", os.path.join(plots_dir, "cm_shape.png"))

    # -------------------------
    # PLOTS (Binned analysis vs speed/omega)
    # -------------------------
    acc_exact_mask = (
        (preds["pred_motion"] == preds["true_motion"])
        & (preds["pred_rot"] == preds["true_rot"])
        & (preds["pred_shape"] == preds["true_shape"])
    )
    save_binned_metrics(
        true_speed=ts_plot,
        pred_speed=ps_plot,
        true_omega=tw_plot,
        pred_omega=pw_plot,
        acc_exact_mask=acc_exact_mask,
        title_prefix=f"{case} | {'TSM' if args.use_tsm else 'noTSM'}",
        out_dir=os.path.join(plots_dir, "binned"),
        num_bins=6,
    )

    print(f"\nSaved plots to: {plots_dir}")
    print(f"Saved test_metrics.json + test_preds.csv to: {out_dir}")


if __name__ == "__main__":
    main()