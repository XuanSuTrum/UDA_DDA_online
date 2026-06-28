"""Command-line entry for quasi-online UDA-DDA new-user adaptation."""

from __future__ import annotations

import argparse
import random
import time
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import cmmd
import mmd
from SDA_DDA import Transfer_Net
from metrics_utils import compute_summary_metrics
from online_data_utils import (
    CLASS_NAMES,
    LABEL_MAP,
    TrialData,
    build_target_trial_stream,
    fit_source_standardizer,
    generate_synthetic_subject_trials,
    load_seed_de_mat_subject_trials,
    load_seed_raw_subject_trials,
    make_source_loader,
    make_target_buffer_loader,
    split_calibration_and_test_trials,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UDA-DDA quasi-online new-user adaptation")
    parser.add_argument("--raw_data_dir", type=str, default=None, help="Directory with raw SEED subject .mat files.")
    parser.add_argument("--de_data_dir", type=str, default=None, help="Directory with processed DE .mat files.")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--input_type", type=str, default="raw", choices=["raw", "de_mat"])
    parser.add_argument("--fs", type=int, default=200)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument("--baseline_seconds", type=float, default=63.0)
    parser.add_argument("--window_size", type=float, default=1.0)
    parser.add_argument("--target_subjects", type=int, nargs="*", default=None)
    parser.add_argument("--protocols", type=str, nargs="+", default=["S0", "S1", "S2"], choices=["S0", "S1", "S2"])
    parser.add_argument("--k_list", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--source_epochs", type=int, default=50)
    parser.add_argument("--cal_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=1.0, help="Target CE weight for S2.")
    parser.add_argument("--beta", type=float, default=1.0, help="MMD weight.")
    parser.add_argument("--lambda_cmmd", type=float, default=1.0, help="CMMD weight.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_net", type=str, default="CFE")
    parser.add_argument("--classifier_width", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Run an end-to-end synthetic-data smoke test.")
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "predictions": output_dir / "predictions",
        "summaries": output_dir / "summaries",
        "checkpoints": output_dir / "checkpoints",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def create_model(args: argparse.Namespace, device: torch.device) -> Transfer_Net:
    model = Transfer_Net(
        num_class=3,
        base_net=args.base_net,
        transfer_loss="mmd",
        width=args.classifier_width,
        confidence_threshold=args.tau,
    )
    return model.to(device)


def train_source_only(
    model: Transfer_Net,
    source_loader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        losses = []
        for x_source, y_source in source_loader:
            if x_source.size(0) < 2:
                continue
            x_source = x_source.to(device)
            y_source = y_source.to(device)
            optimizer.zero_grad()
            logits = model(x_source)
            loss = criterion(logits, y_source)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epochs <= 5 or (epoch + 1) % max(epochs // 5, 1) == 0:
            avg = np.mean(losses) if losses else 0.0
            print(f"  source epoch {epoch + 1}/{epochs}: ce={avg:.4f}")
    return optimizer


def adapt_model(
    model: Transfer_Net,
    source_loader,
    target_loader,
    protocol: str,
    epochs: int,
    lr: float,
    device: torch.device,
    alpha: float,
    beta: float,
    lambda_cmmd: float,
    tau: float,
) -> Tuple[torch.optim.Optimizer, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    target_batches = list(target_loader)
    if not target_batches:
        raise ValueError("target buffer loader is empty")

    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        target_iter = cycle(target_batches)
        losses = []
        confident_counts = []
        for x_source, y_source in source_loader:
            x_target, y_target = next(target_iter)
            if x_source.size(0) < 2 or x_target.size(0) < 2:
                continue

            x_source = x_source.to(device)
            y_source = y_source.to(device)
            x_target = x_target.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()
            source_logits, source_features = model(x_source, return_features=True)
            target_logits, target_features = model(x_target, return_features=True)

            source_ce = criterion(source_logits, y_source)
            mmd_loss = mmd.mmd_rbf_noaccelerate(source_features, target_features)

            if protocol == "S1":
                probs = F.softmax(target_logits.detach(), dim=1)
                confidence, pseudo_y = probs.max(dim=1)
                mask = confidence > tau
                confident_counts.append(int(mask.sum().item()))
                cmmd_loss = cmmd.cmmd(
                    source_features,
                    target_features[mask],
                    y_source,
                    pseudo_y[mask],
                    num_classes=3,
                )
                target_ce = source_features.new_tensor(0.0)
            elif protocol == "S2":
                target_ce = criterion(target_logits, y_target)
                cmmd_loss = cmmd.cmmd(source_features, target_features, y_source, y_target, num_classes=3)
            else:
                raise ValueError(f"unsupported adaptation protocol: {protocol}")

            loss = source_ce + alpha * target_ce + beta * mmd_loss + lambda_cmmd * cmmd_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if epochs <= 5 or (epoch + 1) % max(epochs // 5, 1) == 0:
            msg = f"  {protocol} cal epoch {epoch + 1}/{epochs}: loss={np.mean(losses) if losses else 0.0:.4f}"
            if protocol == "S1" and confident_counts:
                msg += f", confident_avg={np.mean(confident_counts):.1f}"
            print(msg)
    return optimizer, time.perf_counter() - started


def checkpoint_payload(
    model: Transfer_Net,
    optimizer: Optional[torch.optim.Optimizer],
    args: argparse.Namespace,
    source_subjects: Sequence[int],
    target_subject: int,
    standardizer,
) -> dict:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model.model_config(),
        "feature_dim": 310,
        "class_num": 3,
        "label_map": LABEL_MAP,
        "class_names": CLASS_NAMES,
        "seed": args.seed,
        "source_subjects": list(source_subjects),
        "target_subject": target_subject,
        "standardizer_mean": standardizer.scaler.mean_.astype(np.float32),
        "standardizer_scale": standardizer.scaler.scale_.astype(np.float32),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def save_checkpoint(
    path: Path,
    model: Transfer_Net,
    optimizer: Optional[torch.optim.Optimizer],
    args: argparse.Namespace,
    source_subjects: Sequence[int],
    target_subject: int,
    standardizer,
) -> None:
    torch.save(checkpoint_payload(model, optimizer, args, source_subjects, target_subject, standardizer), path)


def load_model_checkpoint(path: Path, args: argparse.Namespace, device: torch.device) -> Transfer_Net:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    config = payload.get("model_config", {})
    model = Transfer_Net(
        num_class=int(config.get("num_class", 3)),
        base_net=config.get("base_net", args.base_net),
        transfer_loss=config.get("transfer_loss", "mmd"),
        width=int(config.get("width", args.classifier_width)),
        confidence_threshold=float(config.get("confidence_threshold", args.tau)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model


def evaluate_stream(
    model: Transfer_Net,
    subject_id: int,
    protocol: str,
    k_value: int,
    test_trials: Sequence[TrialData],
    device: torch.device,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    model.eval()
    records = []
    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    infer_seconds = 0.0
    window_count = 0

    with torch.no_grad():
        for trial in build_target_trial_stream(test_trials):
            if trial.feature.shape[0] == 0:
                continue
            x = torch.from_numpy(trial.feature.astype(np.float32)).to(device)
            started = time.perf_counter()
            logits = model.predict(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            infer_seconds += time.perf_counter() - started

            preds = probs.argmax(axis=1)
            conf = probs.max(axis=1)
            for window_id, (prob, pred, confidence) in enumerate(zip(probs, preds, conf)):
                y_true = int(trial.window_labels[window_id])
                correct = int(pred == y_true)
                records.append(
                    {
                        "subject_id": subject_id,
                        "protocol": protocol,
                        "K": k_value,
                        "trial_id": trial.trial_id,
                        "window_id": window_id,
                        "y_true": y_true,
                        "y_pred": int(pred),
                        "p_negative": float(prob[0]),
                        "p_neutral": float(prob[1]),
                        "p_positive": float(prob[2]),
                        "p_nonnegative": float(prob[1] + prob[2]),
                        "confidence": float(confidence),
                        "correct": correct,
                    }
                )
                y_true_all.append(y_true)
                y_pred_all.append(int(pred))
            window_count += trial.feature.shape[0]

    metrics = compute_summary_metrics(y_true_all, y_pred_all)
    metrics["num_test_windows"] = int(window_count)
    metrics["inference_seconds_per_window"] = float(infer_seconds / max(window_count, 1))
    return pd.DataFrame.from_records(records), metrics


def load_subject_trials(args: argparse.Namespace) -> Dict[int, List[TrialData]]:
    if args.dry_run:
        print("Running dry_run with synthetic trial streams.")
        return generate_synthetic_subject_trials(seed=args.seed)
    if args.input_type == "raw":
        if not args.raw_data_dir:
            raise ValueError("--raw_data_dir is required when --input_type raw")
        return load_seed_raw_subject_trials(
            args.raw_data_dir,
            subject_ids=None,
            fs=args.fs,
            session=args.session,
            baseline_seconds=args.baseline_seconds,
            window_size=args.window_size,
        )
    if not args.de_data_dir:
        raise ValueError("--de_data_dir is required when --input_type de_mat")
    return load_seed_de_mat_subject_trials(args.de_data_dir, subject_ids=None, session=args.session)


def subject_tag(subject_id: int) -> str:
    return f"subject{subject_id:02d}"


def run_experiment(args: argparse.Namespace) -> Path:
    setup_seed(args.seed)
    if args.dry_run:
        if args.source_epochs == 50:
            args.source_epochs = 3
        if args.cal_epochs == 10:
            args.cal_epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    output_paths = ensure_output_dirs(Path(args.output_dir))
    subjects = load_subject_trials(args)
    all_subject_ids = sorted(subjects)
    target_subjects = args.target_subjects or all_subject_ids

    missing = set(target_subjects) - set(all_subject_ids)
    if missing:
        raise ValueError(f"target_subjects not available: {sorted(missing)}")

    summary_records = []
    for target_subject in target_subjects:
        print(f"\nTarget subject {target_subject}")
        source_subjects = [sid for sid in all_subject_ids if sid != target_subject]
        if not source_subjects:
            raise ValueError("at least two subjects are required for leave-one-subject-out training")

        source_trials_raw = [trial for sid in source_subjects for trial in subjects[sid]]
        target_trials_raw = build_target_trial_stream(subjects[target_subject])
        standardizer = fit_source_standardizer(source_trials_raw)
        source_trials = standardizer.transform_trials(source_trials_raw)
        target_trials = standardizer.transform_trials(target_trials_raw)
        source_loader = make_source_loader(source_trials, batch_size=args.batch_size)

        model = create_model(args, device)
        source_optimizer = train_source_only(model, source_loader, args.source_epochs, args.lr, device)
        source_ckpt = output_paths["checkpoints"] / f"{subject_tag(target_subject)}_source_only.pth"
        save_checkpoint(source_ckpt, model, source_optimizer, args, source_subjects, target_subject, standardizer)
        print(f"  saved source checkpoint: {source_ckpt}")

        if "S0" in args.protocols:
            s0_model = load_model_checkpoint(source_ckpt, args, device)
            pred_df, metrics = evaluate_stream(s0_model, target_subject, "S0", 0, target_trials, device)
            pred_path = output_paths["predictions"] / f"uda_dda_predictions_{subject_tag(target_subject)}_S0.csv"
            pred_df.to_csv(pred_path, index=False)
            summary_records.append(
                {
                    "subject_id": target_subject,
                    "protocol": "S0",
                    "K": 0,
                    "num_calibration_trials": 0,
                    "num_test_trials": len(target_trials),
                    **metrics,
                    "calibration_seconds": 0.0,
                    "checkpoint_path": str(source_ckpt),
                }
            )

        for protocol in [p for p in args.protocols if p in {"S1", "S2"}]:
            for k_value in args.k_list:
                calibration_trials, test_trials = split_calibration_and_test_trials(target_trials, k_value)
                target_loader = make_target_buffer_loader(calibration_trials, batch_size=args.batch_size)
                adapted_model = load_model_checkpoint(source_ckpt, args, device)
                adapted_optimizer, cal_seconds = adapt_model(
                    adapted_model,
                    source_loader,
                    target_loader,
                    protocol=protocol,
                    epochs=args.cal_epochs,
                    lr=args.lr,
                    device=device,
                    alpha=args.alpha,
                    beta=args.beta,
                    lambda_cmmd=args.lambda_cmmd,
                    tau=args.tau,
                )
                ckpt_path = output_paths["checkpoints"] / f"{subject_tag(target_subject)}_{protocol}_K{k_value}.pth"
                save_checkpoint(ckpt_path, adapted_model, adapted_optimizer, args, source_subjects, target_subject, standardizer)

                pred_df, metrics = evaluate_stream(adapted_model, target_subject, protocol, k_value, test_trials, device)
                pred_path = (
                    output_paths["predictions"]
                    / f"uda_dda_predictions_{subject_tag(target_subject)}_{protocol}_K{k_value}.csv"
                )
                pred_df.to_csv(pred_path, index=False)
                summary_records.append(
                    {
                        "subject_id": target_subject,
                        "protocol": protocol,
                        "K": k_value,
                        "num_calibration_trials": k_value,
                        "num_test_trials": len(test_trials),
                        **metrics,
                        "calibration_seconds": float(cal_seconds),
                        "checkpoint_path": str(ckpt_path),
                    }
                )

    summary_df = pd.DataFrame.from_records(summary_records)
    summary_path = output_paths["summaries"] / "uda_dda_online_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path}")
    return summary_path


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
