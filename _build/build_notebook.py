import json
import re

with open("12_outlier_analysis.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)


def set_source(idx, text):
    nb["cells"][idx]["source"] = text.splitlines(keepends=True)
    if nb["cells"][idx]["cell_type"] == "code":
        nb["cells"][idx]["execution_count"] = None
        nb["cells"][idx]["outputs"] = []


def patch_source(idx, old, new):
    src = "".join(nb["cells"][idx]["source"])
    assert old in src, f"motif introuvable dans la cellule {idx} : {old[:60]!r}"
    nb["cells"][idx]["source"] = src.replace(old, new).splitlines(keepends=True)


# =============================================================================
# Cellule 0 — introduction
# =============================================================================
set_source(0, """# Notebook 12 — Détection et impact des outliers

## Objectif

Analyser la structure des données, détecter les points aberrants et mesurer l'impact de leur retrait sur les performances du modèle.

**Question centrale :** Y-a-t-il des valeurs aberrantes à retirer ? Si oui, ce retrait est-il scientifiquement justifié ?

## Données
- `radio_features.csv` — mesures avec features + cible, cible : `E_field` (dBµV/m)
- Environnements : Cotonou (urbain, code=1) et Kandi (suburbain, code=0)
- Ce fichier est un export « brut » de 47 colonnes. On ne conserve que les features exploitées par l'ancien notebook (celles de `dataset2_Efield.csv`), **moins `Prcp`** (quasi-constante : 349 valeurs à 0 et 4 à 0.4 mm), soit **22 features** + la cible `E_field`, dans le même ordre ; les autres colonnes (identifiants, coordonnées, altitudes, `Rx_power`, `FSM_spread`, `tx_ratio`, `Wpgt`…) sont écartées.
- `Environment` est ré-encodé en 0/1, et la ligne sans valeur de cible est retirée → **352 mesures**.

## Seuil de nettoyage retenu
**Z < 1.8** sur les résidus de cross-validation (`Z_CLEAN`), soit 25 points retirés → **327 mesures** dans le dataset final.

## Plan
1. EDA complète — distributions, variance, corrélations
2. Détection des outliers — résidus CV, IsolationForest, LOF
3. Analyse physique des clusters d'outliers
4. Impact quantitatif — R² en fonction du seuil de retrait
5. Modèle final sur données nettoyées
6. Conclusion et recommandation
""")

# =============================================================================
# Cellule 2 — imports et configuration
# =============================================================================
set_source(2, """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import math
import os
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

SEED      = 42
BASE      = "."                             # racine du projet (adapter si besoin)
DATA_FILE = f"{BASE}/radio_features.csv"    # ← fichier d'entrée (remplace dataset2_Efield.csv)
TARGET    = "E_field"
Z_CLEAN   = 1.8                             # seuil de nettoyage retenu (|z résidu CV|)

os.makedirs(f"{BASE}/outputs", exist_ok=True)

print("Librairies chargées.")
print(f"Seuil de nettoyage retenu : Z < {Z_CLEAN}")
""")

# =============================================================================
# Cellule 3 — chargement et sélection des colonnes
# =============================================================================
set_source(3, """# ── Chargement + sélection des colonnes de l'ancien notebook ────────────────
# radio_features.csv est un export "brut" de 47 colonnes. On ne garde que les
# features réellement exploitées par l'ancien notebook (= les colonnes de
# dataset2_Efield.csv) MOINS `Prcp` (349 valeurs à 0 et 4 à 0.4 mm :
# quasi-constante, et ses 4 valeurs non nulles créaient de faux extrêmes Z>3).

raw = pd.read_csv(DATA_FILE)

# Encodage numérique de l'environnement (Cotonou/urbain=1, Kandi/suburbain=0)
raw["Environment"] = raw["Environment"].map({"urban": 1, "suburban": 0})

FEATURES = [
    "Temp", "Rhum", "Wdir", "Wspd", "Pres",
    "Distance", "Tx_power", "Tx_height", "Freq",
    "Slope_Tx_Rx_50m", "Roughness_Tx_Rx_50m",
    "Rx_height", "Azimut_Tx_Rx", "Tilt_Tx_Rx", "LOS",
    "First_building_m", "First_tree_m",
    "Fresnel_buildings", "Fresnel_trees",
    "Buildings_near_Rx", "Trees_near_Rx",
    "Environment",
]

n_before = len(raw)
dropped_cols = [c for c in raw.columns if c not in FEATURES + [TARGET]]
df = raw[FEATURES + [TARGET]].copy()

n_missing = df.isna().any(axis=1).sum()
df = df.dropna()   # conserve les index d'origine (utile pour retrouver `Ref`)

ref_lookup = raw.loc[df.index, "Ref"]

print(f"Fichier source : {DATA_FILE}")
print(f"  {len(FEATURES)} features conservées (colonnes de l'ancien notebook, sans Prcp)")
print(f"  {len(dropped_cols)} colonnes écartées : {dropped_cols}")
print(f"  {n_before} lignes brutes → {n_missing} retirées (valeurs manquantes) → {len(df)} conservées")
print(f"Dataset : {df.shape[0]} mesures × {len(FEATURES)} features + cible")
print(f"Cible   : {TARGET} — moy={df[TARGET].mean():.1f}  std={df[TARGET].std():.2f}  "
      f"[{df[TARGET].min():.1f} ; {df[TARGET].max():.1f}] dBµV/m")
print()
print("Répartition par environnement :")
for env, lbl in [(1, "Urban (Cotonou)"), (0, "Rural (Kandi)")]:
    sub = df[df["Environment"] == env][TARGET]
    print(f"  {lbl}: {len(sub)} pts — moy={sub.mean():.1f}  std={sub.std():.2f}  "
          f"[{sub.min():.1f};{sub.max():.1f}] dBµV/m")
""")

# =============================================================================
# Cellule 7 — boxplots, grille dimensionnée automatiquement
# =============================================================================
set_source(7, """# ── 1C. Boxplots des features numériques — distribution et valeurs extrêmes
num_features = [f for f in FEATURES if df[f].nunique() > 5]

ncols = 6
nrows = math.ceil(len(num_features) / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(20, 3 * nrows))
fig.suptitle("Boxplots des variables — distribution et valeurs extrêmes",
             fontsize=12, fontweight="bold")

for ax, feat in zip(axes.flat, num_features):
    data_u = df[df["Environment"] == 1][feat].dropna()
    data_r = df[df["Environment"] == 0][feat].dropna()
    bp = ax.boxplot([data_u, data_r], patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][1].set_facecolor("tomato")
    for el in bp["fliers"]:
        el.set(marker="o", markersize=3, alpha=0.5, color="gray")
    ax.set_title(feat, fontsize=7)
    ax.set_xticklabels(["Urban", "Rural"], fontsize=6)
    ax.tick_params(labelsize=6)

for ax in axes.flat[len(num_features):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(f"{BASE}/outputs/fig12_boxplots_features.png", dpi=120)
plt.show()
print("Fig 1C sauvegardée.")
""")

# =============================================================================
# Cellule 13 — tableau de synthèse : le flag principal suit Z_CLEAN
# =============================================================================
patch_source(13, """df["out_z20"]  = df["z_resid"] > 2.0   # résidu CV Z>2
df["out_z25"]  = df["z_resid"] > 2.5   # résidu CV Z>2.5""",
"""df["out_clean"] = df["z_resid"] > Z_CLEAN   # seuil de nettoyage retenu
df["out_z20"]   = df["z_resid"] > 2.0       # résidu CV Z>2 (comparaison)
df["out_z25"]   = df["z_resid"] > 2.5       # résidu CV Z>2.5 (comparaison)""")

patch_source(13, """df["n_methods"] = (df[["out_z20","out_iso","out_lof"]]).sum(axis=1)""",
"""df["n_methods"] = (df[["out_clean","out_iso","out_lof"]]).sum(axis=1)""")

patch_source(13, """for label, col in [
    ("Z-résidu CV > 2.0σ",           "out_z20"),
    ("Z-résidu CV > 2.5σ",           "out_z25"),""",
"""for label, col in [
    (f"Z-résidu CV > {Z_CLEAN}σ  ← retenu", "out_clean"),
    ("Z-résidu CV > 2.0σ",           "out_z20"),
    ("Z-résidu CV > 2.5σ",           "out_z25"),""")

# =============================================================================
# Cellules 15, 16, 17, 19 — les figures et l'analyse suivent Z_CLEAN
# =============================================================================
patch_source(15, 'out20 = df["out_z20"].values', 'out20 = df["out_clean"].values')
patch_source(15, 'label="Outlier Z>2")', 'label=f"Outlier Z>{Z_CLEAN}")')

patch_source(16, 'label="Outlier (Z>2)")', 'label=f"Outlier (Z>{Z_CLEAN})")')
patch_source(16, 'axes[0].set_title("Outliers résidu CV Z>2")',
                 'axes[0].set_title(f"Outliers résidu CV Z>{Z_CLEAN}")')
patch_source(16, 'label="Outlier Z>2")],', 'label=f"Outlier Z>{Z_CLEAN}")],')

patch_source(17, 'outliers_df = df[df["out_z20"]].copy()', 'outliers_df = df[df["out_clean"]].copy()')
patch_source(17, 'print(f"=== {len(outliers_df)} OUTLIERS (|résidu CV| > 2σ) ===")',
                 'print(f"=== {len(outliers_df)} OUTLIERS (|résidu CV| > {Z_CLEAN}σ) ===")')

patch_source(19, 'normal = ~df["out_z20"].values', 'normal = ~df["out_clean"].values')
patch_source(19, 'out_df = df[df["out_z20"]]', 'out_df = df[df["out_clean"]]')
patch_source(19, 'label="Outlier (|z|>2)")', 'label=f"Outlier (|z|>{Z_CLEAN})")')
patch_source(19, 'ax.set_title("E_field vs Distance — outliers (|z résidu CV| > 2σ) identifiés",',
                 'ax.set_title(f"E_field vs Distance — outliers (|z résidu CV| > {Z_CLEAN}σ) identifiés",')
patch_source(19, 'label="Outlier |z|>2")]', 'label=f"Outlier |z|>{Z_CLEAN}")]')

# =============================================================================
# Cellule 18 — Ref déjà présent dans radio_features.csv
# =============================================================================
set_source(18, """# ── Récupération des références (Ref) des points identifiés comme outliers ──
# radio_features.csv contient déjà la colonne `Ref` ; on la récupère via
# `ref_lookup` (index d'origine préservés), sans recharger de fichier séparé.
idx_out = outliers_df.index.tolist()
print("indices :", idx_out)

refs_out = ref_lookup.loc[idx_out].tolist()
print("refs    :", refs_out)
""")

# =============================================================================
# Cellules 21 et 22 — seuils clés et annotations calculés, non codés en dur
# =============================================================================
patch_source(21, """key_removed = [0, 2, 6, 17, 26, 46]
for _, row in thresh_df.iterrows():
    if int(row["n_removed"]) in key_removed:
        flag = " ←" if abs(row["thresh"] - 2.0) < 0.05 else \"\"""",
"""key_thresholds = [3.0, 2.5, 2.0, 1.8, 1.5]
for _, row in thresh_df.iterrows():
    if round(row["thresh"], 1) in key_thresholds:
        flag = " ← retenu" if abs(row["thresh"] - Z_CLEAN) < 0.05 else \"\"""")

patch_source(22, """for thresh_val, txt in [(2.5, "Z<2.5\\n−6"), (2.0, "Z<2.0\\n−17"),
                         (1.8, "Z<1.8\\n−26"), (1.5, "Z<1.5\\n−46")]:
    row = thresh_df[thresh_df["thresh"].round(1) == round(thresh_val, 1)].iloc[0]""",
"""for thresh_val in [2.5, 2.0, 1.8, 1.5]:
    row = thresh_df[thresh_df["thresh"].round(1) == round(thresh_val, 1)].iloc[0]
    txt = f"Z<{thresh_val}\\n−{int(row['n_removed'])}\"""")

# =============================================================================
# Cellule 23 — introduction de la Section 5
# =============================================================================
set_source(23, """## Section 5 — Modèle final sur données nettoyées

On applique plusieurs seuils et on mesure le gain sur un **test set indépendant** (pas CV), le seuil retenu étant **Z < 1.8**.

> **Note :** le retrait est fait avant le split train/test, donc le test set est lui aussi nettoyé — les valeurs ci-dessous sont comparables entre seuils, et optimistes dans l'absolu.
""")

# =============================================================================
# Cellule 24 — libellés construits à partir des effectifs réels
# =============================================================================
set_source(24, """def eval_cleaned(z_thresh, tag):
    '''Entraîne RF + XGBoost sur les données nettoyées (seuil z_thresh) et retourne les métriques.'''
    mask_clean = z_residuals <= z_thresh
    n_kept = int(mask_clean.sum())
    label = f"{tag} ({n_kept} pts)"
    df_clean = df[mask_clean].reset_index(drop=True)

    X_c = df_clean[FEATURES].values
    y_c = df_clean[TARGET].values

    # Split stratifié par environnement
    train_c, test_c = train_test_split(
        df_clean, test_size=0.20, random_state=SEED, stratify=df_clean["Environment"]
    )
    X_tr = train_c[FEATURES].values; y_tr = train_c[TARGET].values
    X_te = test_c[FEATURES].values;  y_te = test_c[TARGET].values

    # RF
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    yp_rf = rf.predict(X_te)

    # XGBoost
    xgb_m = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=SEED, verbosity=0, n_jobs=-1)
    xgb_m.fit(X_tr, y_tr)
    yp_xgb = xgb_m.predict(X_te)

    rows = []
    tu = test_c["Environment"].values == 1
    tr = test_c["Environment"].values == 0

    for mod_lbl, yp in [("RF", yp_rf), ("XGBoost", yp_xgb)]:
        for env_lbl, emask in [("Global", np.ones(len(y_te), bool)), ("Urban", tu), ("Rural", tr)]:
            rows.append({
                "Seuil": label,
                "N_train": len(train_c), "N_test": len(test_c),
                "Modèle": mod_lbl, "Env": env_lbl,
                "RMSE": round(float(np.sqrt(mean_squared_error(y_te[emask], yp[emask]))), 3),
                "R2":   round(float(r2_score(y_te[emask], yp[emask])), 3),
                "Bias": round(float(np.mean(yp[emask] - y_te[emask])), 3),
            })

    res = pd.DataFrame(rows)
    for mod in ["RF", "XGBoost"]:
        g = res[(res["Modèle"]==mod) & (res["Env"]=="Global")].iloc[0]
        u = res[(res["Modèle"]==mod) & (res["Env"]=="Urban")].iloc[0]
        r = res[(res["Modèle"]==mod) & (res["Env"]=="Rural")].iloc[0]
        print(f"  {label} | {mod:<8} — Global: RMSE={g['RMSE']:.2f} R²={g['R2']:.3f}  "
              f"Urban: RMSE={u['RMSE']:.2f}  Rural: RMSE={r['RMSE']:.2f}")
    return res

print("=" * 72)
print("Baseline (données complètes) :")
res_base = eval_cleaned(99.0, "Complet")
print()
print("Nettoyage conservateur (Z<2.5) :")
res_z25 = eval_cleaned(2.5, "Z<2.5")
print()
print("Nettoyage standard (Z<2.0) :")
res_z20 = eval_cleaned(2.0, "Z<2.0")
print()
print(f"Nettoyage retenu (Z<{Z_CLEAN}) :")
res_z18 = eval_cleaned(Z_CLEAN, f"Z<{Z_CLEAN}")
print()
print("Nettoyage étendu (Z<1.5) :")
res_z15 = eval_cleaned(1.5, "Z<1.5")
print("=" * 72)
""")

# =============================================================================
# Cellule 25 — barplot, libellés issus des résultats
# =============================================================================
patch_source(25, """seuils   = ["Complet (355 pts)", "Z<2.5 (349 pts)", "Z<2.0 (338 pts)",
            "Z<1.8 (329 pts)", "Z<1.5 (309 pts)"]""",
"""seuils   = [res["Seuil"].iloc[0] for res in [res_base, res_z25, res_z20, res_z18, res_z15]]""")

# =============================================================================
# Cellule 26 — scatter du modèle nettoyé au seuil retenu
# =============================================================================
patch_source(26, """# ── Scatter plot : meilleur modèle nettoyé (Z<2.0) ──────────────────────────
df_clean = df[z_residuals <= 2.0].reset_index(drop=True)""",
"""# ── Scatter plot : modèle nettoyé au seuil retenu ───────────────────────────
df_clean = df[z_residuals <= Z_CLEAN].reset_index(drop=True)""")

patch_source(26, """fig.suptitle("RF sur données nettoyées (Z<2.0, -17 pts) — Test set 20%",
             fontsize=11, fontweight="bold")""",
"""n_removed_clean = int((z_residuals > Z_CLEAN).sum())
fig.suptitle(f"RF sur données nettoyées (Z<{Z_CLEAN}, -{n_removed_clean} pts, "
             f"{len(df_clean)} pts) — Test set 20%",
             fontsize=11, fontweight="bold")""")

# =============================================================================
# Cellule 28 — export du dataset nettoyé au seuil retenu
# =============================================================================
set_source(28, """# ── Sauvegarder le dataset nettoyé ──────────────────────────────────────────
df_export = df[z_residuals <= Z_CLEAN][[
    c for c in df.columns
    if c not in ["resid_cv", "z_resid", "iso_flag", "iso_score",
                 "lof_flag", "out_clean", "out_z20", "out_z25", "out_iso",
                 "out_lof", "n_methods", "out_consensus"]
]].reset_index(drop=True)

out_path = f"{BASE}/outputs/radio_features_clean.csv"
df_export.to_csv(out_path, index=False)

print(f"Dataset nettoyé sauvegardé : {out_path}   (seuil Z<{Z_CLEAN})")
print(f"  Shape : {df_export.shape}")
print(f"  Colonnes : {list(df_export.columns)}")
print(f"  E_field  : moy={df_export[TARGET].mean():.1f}  std={df_export[TARGET].std():.2f}  "
      f"[{df_export[TARGET].min():.1f};{df_export[TARGET].max():.1f}]")
print(f"  Urban : {(df_export['Environment']==1).sum()} pts")
print(f"  Rural : {(df_export['Environment']==0).sum()} pts")
""")

# =============================================================================
# Rural → Suburbain partout (aucun identifiant ne contient "rural", vérifié)
# =============================================================================
for c in nb["cells"]:
    src = "".join(c["source"])
    src = src.replace("Rural", "Suburbain").replace("rural", "suburbain")
    c["source"] = src.splitlines(keepends=True)

for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []

with open("12_outlier_analysis_radio_features.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"written — {len(nb['cells'])} cellules")
