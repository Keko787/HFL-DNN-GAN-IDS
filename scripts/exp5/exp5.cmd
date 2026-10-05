@echo off
rem Exp 5: every stage, its levers and its reports, through scripts\exp5\launch.py.
rem
rem   scripts\exp5\exp5 check                  is this machine ready; what each stage needs
rem   scripts\exp5\exp5 run pilots             the pilots, stopping at each report
rem   scripts\exp5\exp5 report knee --apply    write a finished pilot's outputs to params.toml
rem   scripts\exp5\exp5 run batch1 --yes       a stage (or a group: pilots, rl, batches, all)
rem   scripts\exp5\exp5 score batch1           the studies' paired comparisons
rem   scripts\exp5\exp5 status                 every stage's progress
rem   scripts\exp5\exp5 --help                 every command, stage and lever
rem
rem The interpreter: EXP5_PYTHON if set (the conda environment's python.exe, or a
rem .cmd that starts it), else ..\py311.cmd beside the repository if it exists,
rem else python on PATH. Not a venv's python.exe: on Windows that is a launcher
rem stub that doubles every process (the launcher refuses it).
setlocal
set "HERE=%~dp0"
if defined EXP5_PYTHON (
    set "PY=%EXP5_PYTHON%"
) else if exist "%HERE%..\..\..\py311.cmd" (
    set "PY=%HERE%..\..\..\py311.cmd"
) else (
    set "PY=python"
)
call "%PY%" "%HERE%launch.py" %*
exit /b %ERRORLEVEL%
