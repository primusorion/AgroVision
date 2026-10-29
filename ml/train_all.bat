@echo off
rem Train every leaf model for the comparison, then pick the deployed one and rebuild report, samples and checks.
rem Safe to stop at any time (close this window): run it again and each model resumes from its last finished epoch;
rem models whose results\leaf\<arch>\metrics.json already exists are skipped.
cd /d "%~dp0.."
set PY=python
if exist .venv\Scripts\python.exe set PY=.venv\Scripts\python.exe

call :train efficientnet_v2_s 10 || goto :fail
call :train convnext_tiny 10 || goto :fail
call :train mobilenet_v3_large 10 || goto :fail
call :train baseline_cnn 12 --lr 2e-3 || goto :fail

echo [%time%] report (picks the deployed model on validation F1), samples, end-to-end checks
%PY% -u ml\report.py > results\report.log 2>&1 || goto :fail
%PY% -u ml\make_samples.py > results\make_samples.log 2>&1 || goto :fail
%PY% -u ml\pipeline_eval.py > results\pipeline_eval.log 2>&1 || goto :fail
%PY% -u backend\test_api.py > results\test_api.log 2>&1 || goto :fail
echo done > results\train_all.done
echo [%time%] all done
exit /b 0

:train
if exist results\leaf\%1\metrics.json (echo skip %1 - already trained& exit /b 0)
echo [%time%] training %1 - log: results\train_leaf_%1.log
%PY% -u ml\train.py --task leaf --arch %1 --epochs %2 %3 %4 --resume >> results\train_leaf_%1.log 2>&1 || exit /b 1
%PY% -u ml\calibrate.py --task leaf --arch %1 > results\calibrate_leaf_%1.log 2>&1 || exit /b 1
exit /b 0

:fail
echo FAILED - see the newest log in results\
exit /b 1
