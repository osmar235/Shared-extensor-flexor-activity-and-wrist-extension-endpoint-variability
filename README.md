# Shared extensor-flexor activity and wrist-extension endpoint variability

Code and derived outputs for the manuscript on shared activity between extensor carpi ulnaris (ECU) and flexor carpi radialis (FCR) during unilateral wrist extension.

## Study and scope

Secondary analysis of the open EEG-EMG-kinematics wrist-pointing dataset (45 participants). The movement endpoint outcome is **run-level normalized endpoint dispersion**, not error relative to a target. Analyses distinguish between-participant associations (three Spearman tests, Holm correction) and a within-participant regression with participant-clustered standard errors. The study does **not** establish common neural drive, electrical crosstalk, or a causal mechanism.

Original dataset: Perez-Blanco et al. (2026), *Scientific Data*, DOI: [10.1038/s41597-026-07287-z](https://doi.org/10.1038/s41597-026-07287-z). Dataset files: [Figshare v3](https://doi.org/10.6084/m9.figshare.29666735.v3).

## Reproducibility status

**Repository initialized; complete source-code/data deposit is being finalized.** The computational work used scripts named `paper1_common_drive45.py` and `paper1_final_motion_association.py` with participant/run-level CSVs. Please do not interpret this README as a claim that an end-to-end rerun is possible until the source files, their dependencies, quality-control mask, analysis environment and data paths have all been uploaded and checked.

The original raw EDF and wrist-motion files are not redistributed; download them from the official dataset and respect its license and citation requirements.

## Reported results (for comparison)

- 45 participants and 448 EDF runs; 357 run-matched recordings included in the final kinematic analysis.
- Between-participant extension associations with normalized endpoint variability: envelope correlation rho=0.184 (Holm p=0.452), excess beta coherence rho=0.183 (Holm p=0.452), and absolute zero-lag correlation rho=0.255 (Holm p=0.273).
- Secondary within-participant envelope-correlation coefficient: +0.001193 in centered log(1+V) per within-person predictor SD (95% CI +0.000539 to +0.001847; p=0.00035).
- The secondary finding followed earlier exploratory analyses in the same dataset; it is not independent confirmation.

## Contact and citation

Author: Osmar Pinto Neto. Manuscript bibliographic details will be completed after acceptance and publication.

**Version note:** Commit hashes or release tags should be cited once the complete reproducibility package has been deposited. A GitHub URL alone is not a permanent DOI. For a citable archived version, link a tagged release to Zenodo and use its DOI after it is issued.
