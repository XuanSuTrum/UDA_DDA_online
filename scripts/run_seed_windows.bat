@echo off
setlocal EnableExtensions

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
pushd "%REPO_ROOT%" || exit /b 1

if not exist scripts\windows_config.local.bat (
  echo Missing scripts\windows_config.local.bat
  echo Copy or create it, then edit DATA_DIR and experiment settings.
  popd
  exit /b 1
)

call scripts\windows_config.local.bat

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i

if not exist "%OUT_DIR%\summaries" mkdir "%OUT_DIR%\summaries"
if not exist "%OUT_DIR%\figures" mkdir "%OUT_DIR%\figures"
if not exist "%OUT_DIR%\logs" mkdir "%OUT_DIR%\logs"
if not exist "%OUT_DIR%\predictions" mkdir "%OUT_DIR%\predictions"
if not exist checkpoints mkdir checkpoints

set LOG_FILE=%OUT_DIR%\logs\run_%TS%.log
echo Running SEED experiment at %DATE% %TIME% > "%LOG_FILE%"
echo DATA_DIR=%DATA_DIR% >> "%LOG_FILE%"

python online_uda_dda.py ^
  --raw_data_dir "%DATA_DIR%" ^
  --output_dir "%OUT_DIR%" ^
  --input_type raw ^
  --fs %FS% ^
  --target_subjects %TARGET_SUBJECTS% ^
  --protocols %PROTOCOLS% ^
  --k_list %K_LIST% ^
  --source_epochs %SOURCE_EPOCHS% ^
  --cal_epochs %CAL_EPOCHS% ^
  --batch_size %BATCH_SIZE% ^
  --lr %LR% ^
  --tau %TAU% ^
  --seed %SEED% >> "%LOG_FILE%" 2>&1
set RUN_EXIT=%ERRORLEVEL%

if not "%RUN_EXIT%"=="0" (
  echo SEED experiment failed. See %LOG_FILE%
  popd
  exit /b %RUN_EXIT%
)

echo SEED experiment succeeded. Log: %LOG_FILE%
popd
exit /b 0
