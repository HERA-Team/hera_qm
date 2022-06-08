# CHANGELOG
All notable changes to this project will be documented in this file.

## 02-2021

Changes metric approach for correlations and cross_polarizations. Memo in draft,
will be linked here when posted.

- This is a breaking change which removes many functions
- Remove meanvij_metrics, antpol_metric_sum_ratio, per_antenna_modified_z_scores, mean_vij_cross_pol_metrics
- Add cross_pol metrics, calc_corr_stats
- Many changes to AntennaMetrics object and hidden functions
- Many test changes as necessary
