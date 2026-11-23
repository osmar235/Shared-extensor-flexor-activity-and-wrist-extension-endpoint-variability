# Python analysis environment

These scripts are provided as research source code. The original public EDF and motion files are not redistributed.

Install in a dedicated virtual environment:

```
python -m pip install numpy pandas scipy statsmodels mne matplotlib
```

Pipeline:

1. Download public wrist-pointing EDF and motion files under `DATA/sub-XX/...` from https://doi.org/10.6084/m9.figshare.29666735.v3.
2. Place the QC mask `channel_epoch_qc_mask.csv`, and the output inputs `coherence_direction_aggregated.csv` and `coherence_performance_45_subjects.csv` in the working directory.
3. Run `python scripts/paper1_common_drive45.py --data-dir DATA --qc-mask channel_epoch_qc_mask.csv --coherence-csv coherence_direction_aggregated.csv --performance-csv coherence_performance_45_subjects.csv --out-dir RESULTS_COMMON_DRIVE45 --permutations 99`.
4. Run `python scripts/paper1_final_motion_association.py --data-dir DATA --common-drive-csv RESULTS_COMMON_DRIVE45/run_muscle_pair_metrics.csv --cmc-csv run_direction_coherence.csv --out-dir RESULTS_FINAL_MOTION45`.
5. Compare new outputs to the reference CSVs and JSON files in `results/`.

The stage 2 input `run_direction_coherence.csv` is a derived table from a separate upstream EEG–EMG coherence workflow and should be deposited together with the final package. A run-level table by itself does not make that upstream analysis reproducible.

Note that raw files and the QC mask must be available to rerun the full analysis. No code or dataset has been replaced with synthetic results.
