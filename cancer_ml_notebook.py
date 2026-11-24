#!/usr/bin/env python
# coding: utf-8

#  diagnostic du cancer du sein (Breast Cancer Wisconsin)
# 

# ## 1. Imports et configuration

# In[ ]:


import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, accuracy_score, classification_report,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET_RECALL = 0.97
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
for d in (MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True)


# ## 2. Chargement des données
# Dans scikit-learn, `target` vaut 0 = maligne et 1 = bénigne. On inverse pour que
# **1 = maligne** (la classe qu'on veut détecter).

# In[ ]:


raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
df["malignant"] = 1 - df["target"]
df = df.drop(columns="target")
print(df.shape)
df.head()


# ## 3. Nettoyage

# In[ ]:


print("Valeurs manquantes :", df.isna().sum().sum())
print("Doublons :", df.duplicated().sum())
df = df.drop_duplicates().dropna().copy()
df["malignant"].value_counts(normalize=True).round(3)


# ## 4. Analyse exploratoire

# In[ ]:


df.describe().T.head(10)


# In[ ]:


fig, ax = plt.subplots(figsize=(4, 3))
df["malignant"].value_counts().sort_index().plot.bar(ax=ax, color=["#4f81bd", "#c0504d"])
ax.set_title("Bénigne (0) / Maligne (1)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "class_balance.png", dpi=120)
plt.show()


# In[ ]:


corr = df.corr()["malignant"].drop("malignant").sort_values()
plt.figure(figsize=(7, 8))
plt.barh(corr.index, corr.values, color="#4f81bd")
plt.title("Corrélation de chaque variable avec 'malignant'")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "correlation_target.png", dpi=120)
plt.show()


# ## 5. Séparation train / test

# In[ ]:


X = df.drop(columns="malignant")
y = df["malignant"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print("Train :", X_train.shape, "| Test :", X_test.shape)


# ## 6. Pipeline et modèles candidats
# La standardisation est dans le `Pipeline` : elle est recalculée à chaque pli de
# validation croisée, ce qui évite toute fuite de données.

# In[ ]:


def build_pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])

candidates = {
    "LogisticRegression": LogisticRegression(max_iter=5000, random_state=SEED),
    "KNN": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# ## 7. Comparaison par validation croisée (F1, rappel, ROC-AUC)

# In[ ]:


scoring = {"f1": "f1", "recall": "recall", "roc_auc": "roc_auc"}
cv_scores = {}
for name, model in candidates.items():
    r = cross_validate(build_pipeline(model), X_train, y_train,
                       cv=cv, scoring=scoring, n_jobs=-1)
    cv_scores[name] = {k: round(float(r[f"test_{k}"].mean()), 4) for k in scoring}

pd.DataFrame(cv_scores).T.sort_values("f1", ascending=False)


# In[ ]:


best_name = max(cv_scores, key=lambda n: cv_scores[n]["f1"])
print("Meilleur modèle :", best_name)


# ## 8. Optimisation des hyperparamètres (GridSearchCV)

# In[ ]:


param_grids = {
    "LogisticRegression": {"model__C": [0.01, 0.1, 1, 10]},
    "KNN": {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]},
    "RandomForest": {"model__n_estimators": [200, 400],
                     "model__max_depth": [None, 10],
                     "model__min_samples_leaf": [1, 2]},
    "GradientBoosting": {"model__n_estimators": [100, 200],
                         "model__learning_rate": [0.05, 0.1],
                         "model__max_depth": [2, 3]},
}

grid = GridSearchCV(build_pipeline(candidates[best_name]), param_grids[best_name],
                    cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Meilleurs paramètres :", grid.best_params_)
print("F1 (CV) :", round(grid.best_score_, 4))


# ## 9. Choix du seuil de décision
# Par défaut un modèle classe « maligne » à partir de 0,5 de probabilité. Ici on cherche
# le seuil **le plus élevé qui garantit un rappel ≥ 97 %** (en validation croisée sur le
# train uniquement, pour ne pas toucher au jeu de test).

# In[ ]:


proba_cv = cross_val_predict(best_model, X_train, y_train, cv=cv,
                             method="predict_proba", n_jobs=-1)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_train, proba_cv)

ok = np.where(recall[:-1] >= TARGET_RECALL)[0]
threshold = float(thresholds[ok[-1]]) if len(ok) else 0.5
print(f"Seuil retenu : {threshold:.3f}")


# In[ ]:


plt.figure(figsize=(6, 4))
plt.plot(thresholds, precision[:-1], label="précision")
plt.plot(thresholds, recall[:-1], label="rappel")
plt.axvline(threshold, color="gray", linestyle="--", label="seuil retenu")
plt.xlabel("seuil")
plt.legend()
plt.title("Précision et rappel selon le seuil (validation croisée)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "threshold.png", dpi=120)
plt.show()


# ## 10. Évaluation sur le jeu de test

# In[ ]:


proba = best_model.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)

metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, proba), 4),
}
print(metrics)
print(classification_report(y_test, pred, target_names=["bénigne", "maligne"]))


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=axes[0], colorbar=False,
                                        display_labels=["bénigne", "maligne"])
axes[0].set_title("Matrice de confusion")
RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1])
axes[1].set_title("Courbe ROC")
PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[2])
axes[2].set_title("Courbe précision / rappel")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "evaluation.png", dpi=120)
plt.show()


# ## 11. Importance des variables (par permutation)

# In[ ]:


r = permutation_importance(best_model, X_test, y_test, scoring="roc_auc",
                           n_repeats=10, random_state=SEED, n_jobs=-1)
imp = pd.Series(r.importances_mean, index=X_test.columns).nlargest(15).sort_values()

plt.figure(figsize=(7, 5))
plt.barh(imp.index, imp.values, color="#4f81bd")
plt.title("Top 15 des variables (importance par permutation)")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=120)
plt.show()


# ## 12. Sauvegarde du modèle, du seuil et des métriques

# In[ ]:


joblib.dump({"model": best_model, "threshold": threshold,
             "features": list(X.columns)}, MODELS_DIR / "model.pkl")

report = {"best_model": best_name, "best_params": grid.best_params_,
          "threshold": threshold, "cv_scores": cv_scores, "test_metrics": metrics}
(MODELS_DIR / "metrics.json").write_text(json.dumps(report, indent=2, default=str))
print("Sauvegardé dans", MODELS_DIR.resolve())


# ## 13. Prédiction sur de nouveaux patients

# In[ ]:


bundle = joblib.load(MODELS_DIR / "model.pkl")
sample = df.sample(3, random_state=1)
p = bundle["model"].predict_proba(sample[bundle["features"]])[:, 1]

for (_, row), prob in zip(sample.iterrows(), p):
    verdict = "MALIGNE" if prob >= bundle["threshold"] else "bénigne"
    reel = "maligne" if row["malignant"] == 1 else "bénigne"
    print(f"P(maligne) = {prob:6.1%} -> {verdict:8s} (réel : {reel})")

