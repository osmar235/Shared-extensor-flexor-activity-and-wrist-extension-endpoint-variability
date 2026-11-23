@echo off
setlocal
cd /d "%~dp0"
if not exist "DATA" (
  echo ERROR: Original public EDF and kinematic dataset must be placed under DATA.
  exit /b 1
)
python scripts\paper1_common_drive45.py --data-dir DATA --qc-mask qc\channel_epoch_qc_mask.csv --coherence-csv results\coherence_direction_aggregated.csv --performance-csv results\coherence_performance_45_subjects.csv --out-dir RESULTS_COMMON_DRIVE45 --permutations 99
if errorlevel 1 exit /b 1
python scripts\paper1_final_motion_association.py --data-dir DATA --common-drive-csv RESULTS_COMMON_DRIVE45\run_muscle_pair_metrics.csv --cmc-csv results\run_direction_coherence.csv --out-dir RESULTS_FINAL_MOTION45
if errorlevel 1 exit /b 1
echo Completed. Compare outputs with original results folder.
endlocal
