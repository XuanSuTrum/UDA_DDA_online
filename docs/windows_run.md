# Windows Run Guide

This guide assumes the local repository root is:

```bat
E:\show\online-transfer\uda-dda-online
```

## 1. Open a Terminal

Open Anaconda Prompt or PowerShell.

For `cmd`/Anaconda Prompt:

```bat
cd /d E:\show\online-transfer\uda-dda-online
```

For PowerShell:

```powershell
Set-Location E:\show\online-transfer\uda-dda-online
```

## 2. Create Environment

```bat
conda create -n uda_online python=3.9 -y
conda activate uda_online
pip install -r requirements.txt
```

Install the PyTorch CUDA build you need separately if your machine should use GPU.

## 3. Edit Local Config

Edit:

```bat
scripts\windows_config.local.bat
```

At minimum, update:

```bat
set DATA_DIR=F:\Emotion_datasets\preprocess_data\Raw_data\session1
```

This local file is ignored by Git and should not be uploaded.

## 4. Dry Run

Run the synthetic-data smoke test first:

```bat
scripts\run_dry_run_windows.bat
```

The log is written to:

```bat
results\logs\dry_run_YYYYMMDD_HHMMSS.log
```

## 5. Real SEED Run

After confirming `DATA_DIR`, run:

```bat
scripts\run_seed_windows.bat
```

The log is written to:

```bat
results\logs\run_YYYYMMDD_HHMMSS.log
```

If the experiment fails, the script exits with a non-zero code and does not run
any Git operation.

## 6. Run and Upload Results

After you are ready to upload summary artifacts:

```bat
scripts\run_seed_and_push_windows.bat
```

This runs the real SEED experiment, then commits and pushes only:

```bat
results\summaries
results\figures
results\logs
```

It does not upload raw data, `.mat` files, checkpoints, or window-level prediction
CSVs under `results\predictions`.
