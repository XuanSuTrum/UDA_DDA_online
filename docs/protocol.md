# UDA-DDA Quasi-Online New-User Protocol

This project evaluates UDA-DDA under a deployment-like target stream.

## Data Order

Each SEED subject is loaded as ordered trials. Raw EEG trials have shape `(62, T)`.
Each trial is converted independently into DE windows with shape `(n_window, 310)`.
Target trials are never concatenated and shuffled for evaluation.

## S0 Source-Only

The model is trained only on source subjects. Target trials enter in chronological
order and are used only for inference. No target labels or unlabeled target
features are used for adaptation.

## S1 Unlabeled Online Adaptation

For each `K`, the first `K` target trials form an unlabeled calibration buffer.
The source-only model predicts pseudo labels on the buffer. Target windows whose
maximum class probability is greater than `tau` are used in CMMD. The calibration
objective is:

`source CE + beta * MMD(source, target buffer) + lambda_cmmd * CMMD(source, confident target, source labels, pseudo labels)`

Remaining target trials are tested in chronological order.

## S2 Few-Label Calibration

For each `K`, the first `K` target trials and their labels form a small labeled
calibration set. The objective is:

`source CE + alpha * target CE + beta * MMD(source, target buffer) + lambda_cmmd * CMMD(source, target buffer, source labels, target labels)`

Only the first `K` target labels are used for calibration. Remaining target labels
are used only for final metrics.

## Leakage Rules

S1 never reads target labels during calibration. S1 and S2 only use the first `K`
target trials. Future target trials are not used to align, standardize, train, or
select pseudo labels. Source feature standardization is fitted on source subjects
only and then applied to target trials.
