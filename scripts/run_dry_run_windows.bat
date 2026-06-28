@echo off
setlocal EnableExtensions

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
pushd "%REPO_ROOT%" || exit /b 1

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set TS=%%i

if not exist results\summaries mkdir results\summaries
if not exist results\figures mkdir results\figures
if not exist results\logs mkdir results\logs
if not exist results\predictions mkdir results\predictions
if not exist checkpoints mkdir checkpoints

set LOG_FILE=results\logs\dry_run_%TS%.log
echo Running dry_run at %DATE% %TIME% > "%LOG_FILE%"
python online_uda_dda.py --dry_run --output_dir results >> "%LOG_FILE%" 2>&1
set RUN_EXIT=%ERRORLEVEL%

if not "%RUN_EXIT%"=="0" (
  echo dry_run failed. See %LOG_FILE%
  popd
  exit /b %RUN_EXIT%
)

echo dry_run succeeded. Log: %LOG_FILE%
popd
exit /b 0
