# UDA_DDA_online

This repository implements the new-user quasi-online migration learning
experiments for Chapter 5, Section 5.2 of the doctoral dissertation. It studies
how UDA-DDA can be trained, exported, calibrated, and evaluated when a new user's
SEED EEG trials arrive in temporal order.

## Difference From Original UDA-DDA

The original `UDA-DDA` repository is an offline transductive UDA setting: all
unlabeled target-domain data are available during adaptation. This project keeps
target-user trial order. Only the first `K` trials are available for calibration,
and the remaining trials are streamed for testing.

## Protocols

- `S0 Source-only`: train on source subjects only; no target labels and no target
  unlabeled adaptation.
- `S1 Unlabeled online adaptation`: use the first `K` unlabeled target trials,
  pseudo labels, MMD, and high-confidence CMMD.
- `S2 Few-label calibration`: use labels from only the first `K` target trials,
  with target CE plus MMD and CMMD.

Default `K` values are `1 2 3`, configurable with `--k_list`.

## Data Preparation

Raw mode is recommended. Put SEED raw/preprocessed `.mat` files in one directory,
with one subject per file and 15 ordered trial arrays per subject. Each raw trial
should be shaped `(62, T)`; `(T, 62)` is transposed automatically. Default sampling
rate is `fs=200`. For a self-developed acquisition board, set `--fs 250`.

`input_type=de_mat` is also available for processed DE feature files. If a DE
file no longer contains exact trial boundaries, the loader falls back to splitting
the subject feature matrix into 15 contiguous chunks, so raw mode is preferred.

## Installation

On Windows, a conda environment is recommended:

```bat
conda create -n uda-dda-online python=3.10
conda activate uda-dda-online
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build separately if needed.

## Quick Test

The dry run does not require real SEED data. It simulates 3 subjects, 15 trials per
subject, and a small number of windows per trial:

```bat
python online_uda_dda.py --dry_run
```

## Real SEED Session 1 Examples

Windows cmd:

```bat
python online_uda_dda.py ^
  --raw_data_dir "F:\Emotion_datasets\preprocess_data\Raw_data\session1" ^
  --output_dir results ^
  --input_type raw ^
  --fs 200 ^
  --target_subjects 1 2 3 ^
  --protocols S0 S1 S2 ^
  --k_list 1 2 3 ^
  --source_epochs 50 ^
  --cal_epochs 10 ^
  --batch_size 64 ^
  --lr 0.001 ^
  --tau 0.8 ^
  --seed 42
```

PowerShell:

```powershell
python .\online_uda_dda.py `
  --raw_data_dir "F:\Emotion_datasets\preprocess_data\Raw_data\session1" `
  --output_dir results `
  --input_type raw `
  --fs 200 `
  --target_subjects 1 2 3 `
  --protocols S0 S1 S2 `
  --k_list 1 2 3 `
  --source_epochs 50 `
  --cal_epochs 10 `
  --batch_size 64 `
  --lr 0.001 `
  --tau 0.8 `
  --seed 42
```

bash:

```bash
python online_uda_dda.py \
  --raw_data_dir "/path/to/SEED/session1" \
  --output_dir results \
  --input_type raw \
  --fs 200 \
  --target_subjects 1 2 3 \
  --protocols S0 S1 S2 \
  --k_list 1 2 3 \
  --source_epochs 50 \
  --cal_epochs 10 \
  --batch_size 64 \
  --lr 0.001 \
  --tau 0.8 \
  --seed 42
```

## Outputs

Outputs are written under `results/` by default:

- `predictions/uda_dda_predictions_subject01_S0.csv`
- `predictions/uda_dda_predictions_subject01_S1_K1.csv`
- `predictions/uda_dda_predictions_subject01_S2_K1.csv`
- `summaries/uda_dda_online_summary.csv`
- `checkpoints/subject01_source_only.pth`
- `checkpoints/subject01_S1_K1.pth`
- `checkpoints/subject01_S2_K1.pth`

Prediction CSV columns include subject, protocol, `K`, trial/window id, true and
predicted labels, class probabilities, `p_nonnegative`, confidence, and correctness.
The summary CSV includes accuracy, macro F1, negative-class precision/recall/F1,
calibration time, inference time per window, and checkpoint path.

## Data Leakage Rules

- S1 does not use target-domain labels.
- Calibration uses only the first `K` target trials.
- Labels from remaining target trials are used only for final evaluation.
- Future target trials are not used for alignment, pseudo-label filtering, or model
  updates.
- Offline full-target UDA is not implemented in the main flow of this repository.

## Dissertation Section 5.2 Mapping

This project corresponds to the Chapter 5.2 deployment study:

- streaming target-domain adaptation protocol,
- UDA-DDA offline source training, export, and deployment,
- unlabeled pseudo-label alignment,
- few-label supervised calibration,
- continuous negative/neutral/positive and nonnegative emotional-state output.
