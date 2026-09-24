# Pipeline sans fuite de données — DTT path loss, Bénin

Dossier autonome. Décompresser, ouvrir dans VSCode, tout tourne sans rien déplacer : les notebooks, les données et les dossiers de sorties sont au même niveau, et c'est ce que les chemins relatifs attendent.

## Démarrage

```bash
cd dtt-pathloss-leakfree
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
code .
```

Dans VSCode, ouvrir un notebook et sélectionner l'interpréteur `.venv` en haut à droite. Les extensions Python et Jupyter sont suggérées automatiquement. Les sorties de tous les notebooks sont déjà présentes : rien n'a besoin d'être réexécuté pour lire les résultats.

Python 3.11. Les versions de `requirements.txt` sont celles qui ont produit ces résultats.

## Ordre d'exécution

Chaque notebook consomme les sorties du précédent, l'ordre est strict.

| # | Notebook | Rôle | Durée | Écrit dans |
|---|---|---|---|---|
| 12 | `12_outlier_analysis_radio_features.ipynb` | Détection des outliers, seuil Z<1.8 | ~3 min | `outputs/` |
| 07 | `07_leakfree_pipeline.ipynb` | Split Dev/Test, Optuna, importance out-of-fold, Borda, ablation, N\* | **~23 min** | `outputs_nb07/` |
| 08 | `08_hata_hybrides_leakfree.ipynb` | Hata L1a/L1b/L2, ITU-R, hybrides A1/A2 avec ablation N=1..22 | ~10 min | `outputs_nb08/` |
| 09 | `09_hybridation_separation_roles.ipynb` | Ablation sur trois espaces de features pour les hybrides | ~15 min | `outputs_nb09/` |
| 10 | `10_comparaison_standard.ipynb` | Entraînement commun, IC bootstrap, tests appariés, tableau LaTeX | ~1 min | `outputs_nb10/` |

Le 12 régénère `radio_features_clean_ref.csv`, déjà fourni : on peut démarrer au 07. Le 10 se réexécute seul en une minute à partir de `outputs_nb07/` et `outputs_nb08/`.

`_build/` contient les scripts qui ont généré les notebooks — pour la traçabilité, ils ne sont pas nécessaires à l'exécution.

## Données

- `radio_features.csv` — export brut, 353 lignes × 47 colonnes.
- `radio_features_clean_ref.csv` — **entrée du pipeline**. 327 mesures, les 22 prédicteurs, la cible `E_field`, plus `Ref`, `Tx_antenna_height`, `Rx_antenna_height` comme métadonnées.

Écarts avec le dataset de l'article : 327 mesures au lieu de 338 (outliers retirés au seuil Z<1.8) et `Prcp` écartée des prédicteurs, quasi-constante (349 zéros, 4 valeurs à 0.4 mm).

## Correspondance avec le manuscrit

| Élément | Fichier |
|---|---|
| Tableau comparaison inter-familles (LaTeX booktabs) | `outputs_nb10/tableau_manuscrit.tex` |
| Le même en CSV | `outputs_nb10/tableau_manuscrit_IC.csv` |
| Figure comparaison avec barres d'erreur | `outputs_nb10/fig_cross_family_IC.pdf` |
| Tests appariés entre modèles | `outputs_nb10/comparaisons_appariees.csv` |
| Toutes les métriques avec IC | `outputs_nb10/resultats_avec_IC.csv` |
| Classement de Borda | `outputs_nb07/fig_borda_rank_dev.pdf` |
| Importances par modèle | `outputs_nb07/fig_perm_importance_dev.pdf` |
| Ablation du core-set | `outputs_nb07/fig_core_set_ablation_dev.pdf` |
| Choix de N\* (ML pur) | `outputs_nb07/nstar_selection_dev.csv` |
| Ablation des hybrides A1/A2 | `outputs_nb08/hybrid_A{1,2}_ablation_dev.csv`, `fig_ablation_hybrides_dev.pdf` |
| Choix de N\* (hybrides) | `outputs_nb08/nstar_hybrides_dev.csv` |
| Quantification de la fuite, ML pur | `outputs_nb07/comparaison_protocoles.csv` |
| Quantification de la fuite, hybrides | `outputs_nb08/comparaison_protocoles_hybrides.csv` |
| Hyperparamètres retenus | `outputs_nb07/best_params_dev.json` |

## Protocole

Split **80/20 stratifié par `Environment`, `random_state=42`** : Dev 261, Test 66. Vérifié : 64.53 % d'urbain au global, 64.37 % dans Dev, 65.15 % dans le Test. Le filtre physique `Distance ≥ 1000 m` requis par Hata et P.1546-6 est appliqué séparément aux deux partitions : Dev 242, Test 62.

Toutes les décisions sont prises **dans Dev** :

- **Hyperparamètres** — Optuna TPE, 150 essais par modèle, RMSE moyen sur trois hold-out stratifiés tirés dans Dev (seeds 42, 123, 456). Déjà le protocole de l'article.
- **Importance des features** — permutation à 30 répétitions, **out-of-fold** sur 5 plis de Dev, moyennée. L'article la calculait sur le test.
- **Ordre de Borda** — somme des rangs des quatre régresseurs sur ces importances.
- **N\*** — ablation cumulative en validation croisée répétée dans Dev (5 plis × 2 répétitions), puis **règle du plateau à 1 erreur type**. L'article prenait l'`argmin` du RMSE de test.
- **Calibration Hata** — L1a, L1b, L2 estimées sur Dev filtré. Déjà propre dans l'article.
- **Ê_L2 dans l'ablation hybride** — réajusté par OLS dans chaque pli.
- **Modèle représentatif de chaque famille** — désigné par son RMSE de validation dans Dev.

Le test n'est lu qu'une fois, au notebook 10. Les §7 des notebooks 07 et 08 rejouent délibérément l'ancien protocole sur le test, uniquement pour chiffrer la fuite ; rien n'en ressort qui alimente le pipeline.

## Points à traiter dans la rédaction

**La régression L2 n'est pas identifiable.** `statsmodels` émet `SingularMatrixWarning: The design matrix is rank-deficient`. Sept colonnes pour un rang effectif de 5, conditionnement 7·10¹⁶ : `a_hm` est strictement constante (`Rx_antenna_height` = 10 m partout) et `log_f`, `log_erp`, `Environment` ne prennent que deux valeurs déterminées par la ville. Les prédictions Ê_L2 restent valides, donc le RMSE de 7.00 dB tient ; les **valeurs des coefficients** ne sont pas interprétables et ne doivent pas être commentées comme des exposants de propagation ré-estimés. Alternative : re-paramétrer sur `log_d`, `log_hb·log_d`, `Environment` et la constante.

**Le chiffre principal.** Le modèle désigné en validation est CatBoost (5.211 dB dans Dev), qui donne **4.89 dB [4.07, 5.69]** sur le test. Une désignation faite sur le test aurait retenu XGBoost et affiché 4.39 dB — 0.50 dB d'optimisme dû à la seule sélection. Les classements Dev et test sont d'ailleurs différents (Dev : CatBoost, LightGBM, XGBoost, RF ; test : XGBoost, LightGBM, CatBoost, RF).

**Établi statistiquement** (Holm au sein des comparaisons principales, p < 0.05) : le ML bat la référence « moyenne par ville » de 2.30 dB ; L2 bat L1b de 2.93 dB ; L1b bat L1a de 1.64 dB ; L2 bat ITU-R de 6.64 dB.

**Cas limite à discuter** : ML CatBoost vs Hata L2, ΔRMSE = −2.11 dB, IC bootstrap [−3.40, −0.77] qui exclut zéro et P(A<B) = 0.998, mais Wilcoxon+Holm à p = 0.095. Rapporter les deux lectures.

**Non établi** : le classement entre les quatre régresseurs — aucune paire significative. Et l'hybridation n'apporte rien : A1-CatBoost est à −0.08 dB du ML pur, IC [−0.57, +0.40]. Formuler « l'hybridation égale le ML pur », jamais « A1 est le meilleur modèle ».

**Augmentation** non reprise : le notebook 06 d'origine avait établi que le gain CTGAN n'était pas reproductible une fois la graine fixée.
