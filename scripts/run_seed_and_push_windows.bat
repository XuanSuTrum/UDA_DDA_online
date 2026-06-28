@echo off
setlocal EnableExtensions

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
pushd "%REPO_ROOT%" || exit /b 1

call scripts\run_seed_windows.bat
set RUN_EXIT=%ERRORLEVEL%
if not "%RUN_EXIT%"=="0" (
  echo Experiment failed. No git add, commit, or push will be run.
  popd
  exit /b %RUN_EXIT%
)

git add results\summaries results\figures results\logs
git diff --cached --quiet
if "%ERRORLEVEL%"=="0" (
  echo No new results to commit
  popd
  exit /b 0
)

git commit -m "Add UDA-DDA online experiment results"
if not "%ERRORLEVEL%"=="0" (
  echo git commit failed
  popd
  exit /b 1
)

git push origin main
set PUSH_EXIT=%ERRORLEVEL%
if not "%PUSH_EXIT%"=="0" (
  echo git push failed
  popd
  exit /b %PUSH_EXIT%
)

echo Results committed and pushed to origin main.
popd
exit /b 0
