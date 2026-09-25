# Leak-free pipeline — DTT path loss, Benin

Self-contained folder. Unzip, open in VS Code, everything runs without moving anything: notebooks, data, and output folders sit at the same level, matching what the relative paths expect.

## Getting started

```bash
cd dtt-pathloss-leakfree
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
code .
```

In VS Code, open a notebook and select the `.venv` interpreter in the top right. The Python and Jupyter extensions are suggested automatically. Outputs for every notebook are already present: nothing needs to be re-run to read the results.

Python 3.11. The versions pinned in `requirements.txt` are the ones that produced these results.

## Execution order

Each notebook consumes the previous one's outputs; the order is strict.

| # | Notebook | Role | Duration | Writes to |
|---|---|---|---|---|
| 07 | `07_leakfree_pipeline.ipynb` | Dev/Test split, Optuna, out-of-fold importance, Borda, ablation, N\* | **~23 min** | `outputs_nb07/` |
| 08 | `08_hata_hybrides_leakfree.ipynb` | Hata L1a/L1b/L2, ITU-R, hybrids A1/A2 with N=1..19 ablation | ~10 min | `outputs_nb08/` |
| 09 | `09_hybridation_separation_roles.ipynb` | Ablation over three feature spaces for the hybrids | ~15 min | `outputs_nb09/` |
| 10 | `10_comparaison_standard.ipynb` | Common training, bootstrap CI, paired tests, LaTeX table | ~1 min | `outputs_nb10/` |

`radio_features_clean_ref.csv` is already provided: start directly at 07. Notebook 10 reruns on its own in about a minute from `outputs_nb07/` and `outputs_nb08/`.

`_build/` contains the scripts that generated the notebooks — kept for traceability, not needed to run them.

## Data

- `radio_features.csv` — raw export, 353 rows x 47 columns.
- `radio_features_clean_ref.csv` — **pipeline input**. 327 measurements, the 19 predictors, the `E_field` target, plus `Ref`, `Tx_antenna_height`, `Rx_antenna_height` as metadata.

Differences from the article's dataset: 327 measurements instead of 338, and `Prcp` dropped from the predictors as near-constant (349 zeros, 4 values at 0.4 mm).

## Manuscript correspondence

| Item | File |
|---|---|
| Cross-family comparison table (LaTeX booktabs) | `outputs_nb10/tableau_manuscrit.tex` |
| Same table in CSV | `outputs_nb10/tableau_manuscrit_IC.csv` |
| Comparison figure with error bars | `outputs_nb10/fig_cross_family_IC.pdf` |
| Paired tests between models | `outputs_nb10/comparaisons_appariees.csv` |
| All metrics with confidence intervals | `outputs_nb10/resultats_avec_IC.csv` |
| Borda ranking | `outputs_nb07/fig_borda_rank_dev.pdf` |
| Per-model importances | `outputs_nb07/fig_perm_importance_dev.pdf` |
| Core-set ablation | `outputs_nb07/fig_core_set_ablation_dev.pdf` |
| N\* choice (pure ML) | `outputs_nb07/nstar_selection_dev.csv` |
| A1/A2 hybrid ablation | `outputs_nb08/hybrid_A{1,2}_ablation_dev.csv`, `fig_ablation_hybrides_dev.pdf` |
| N\* choice (hybrids) | `outputs_nb08/nstar_hybrides_dev.csv` |
| Leakage quantification, pure ML | `outputs_nb07/comparaison_protocoles.csv` |
| Leakage quantification, hybrids | `outputs_nb08/comparaison_protocoles_hybrides.csv` |
| Retained hyperparameters | `outputs_nb07/best_params_dev.json` |

## Protocol

80/20 split, **stratified by `Environment`, `random_state=42`**: Dev 261, Test 66. Verified: 64.53% urban overall, 64.37% in Dev, 65.15% in Test. The physical filter `Distance >= 1000 m` required by Hata and P.1546-6 is applied separately to both partitions: Dev 242, Test 62.

All decisions are made **within Dev**:

- **Hyperparameters** — Optuna TPE, 150 trials per model, mean RMSE over three stratified hold-outs drawn within Dev (seeds 42, 123, 456). Already the article's protocol.
- **Feature importance** — permutation with 30 repeats, **out-of-fold** over 5 Dev folds, averaged. The article computed it on the test set.
- **Borda order** — sum of ranks across the four regressors on these importances.
- **N\*** — cumulative ablation via repeated cross-validation within Dev (5 folds x 2 repeats), then the **1-standard-error plateau rule**. The article used the test-RMSE argmin.
- **Hata calibration** — L1a, L1b, L2 estimated on filtered Dev. Already clean in the article.
- **Ê_L2 in the hybrid ablation** — refit by OLS within each fold.
- **Representative model per family** — chosen by its Dev validation RMSE.

The test set is read only once, in notebook 10. Section 7 of notebooks 07 and 08 deliberately replays the old protocol on the test set, solely to quantify the leakage; nothing from it feeds the pipeline.

## Open items for the writeup

**The L2 regression is not identifiable.** `statsmodels` raises `SingularMatrixWarning: The design matrix is rank-deficient`. Seven columns for an effective rank of 5, condition number 7e16: `a_hm` is strictly constant (`Rx_antenna_height` = 10 m everywhere) and `log_f`, `log_erp`, `Environment` take only two values, determined by the city. The Ê_L2 predictions remain valid, so the reported RMSE holds; the **coefficient values** are not interpretable and should not be discussed as re-estimated propagation exponents. Alternative: reparameterize on `log_d`, `log_hb·log_d`, `Environment`, and the constant.

**The headline number.** The model selected by Dev validation is LightGBM, which gives **4.88 dB [4.14, 5.57]** on the test set. Test-based selection would instead pick RF, at 4.78 dB [4.11, 5.45] — a modest but real illustration of selection optimism (~0.10 dB here). The best hybrid, A2-CatBoost, gives 4.74 dB [4.02, 5.46] — numerically ahead of every pure-ML model but not distinguishably so (see below).

**Statistically established** (Holm-corrected, within the main comparisons, p < 0.05): calibrated Hata L2 beats L1b by 2.93 dB (p_Holm = 0.029); L2 beats ITU-R P.1546-6 by 6.64 dB (p_Holm < 0.0001). That is the full list — fewer claims hold up than in earlier iterations of this pipeline.

**Not established, despite a favorable point estimate**: ML (LightGBM) vs. Hata L2 — ΔRMSE = -2.12 dB, bootstrap CI [-3.57, -0.59] excluding zero and P(A<B) = 0.997, but Wilcoxon+Holm at p = 0.323. ML vs. the "mean by city" baseline is similarly suggestive but non-significant (ΔRMSE = -2.31 dB, p_Holm = 0.088), as is L1b vs. L1a (ΔRMSE = -1.64 dB, p_Holm = 0.052, the closest miss). Report the point estimate and the p-value together, not the point estimate alone.

**Not established**: the ranking among the four regressors — no pair reaches significance. Hybridization adds nothing measurable either: A2-CatBoost vs. ML LightGBM is -0.14 dB, CI [-0.55, +0.30]. Phrase it as "hybridization matches pure ML," never "the hybrid is the best model."

**Augmentation** not repeated here: the original notebook 06 had already established that the CTGAN gain was not reproducible once the seed was fixed.

*Figures above reflect the current 19-feature pipeline (Wdir, Wspd, and Pres dropped from the weather predictors; Temp and Rhum retained for their link to atmospheric refractivity) and the latest full re-run of notebooks 07-11.*
