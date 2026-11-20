#!/usr/bin/env python3
"""
Pipeline ML complet (Wine Quality) dans un seul fichier.

Usage :
    python pipeline.py            # lance tout : prepare -> train -> evaluate -> test
    python pipeline.py prepare    # une seule étape
    python pipeline.py train
    python pipeline.py evaluate
    python pipeline.py test

Dépendances : pandas, scikit-learn, joblib
    pip install pandas scikit-learn joblib
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- chemins
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
RAW_CSV = DATA / "winequality-red.csv"
TRAIN_CSV = DATA / "train.csv"
TEST_CSV = DATA / "test.csv"
MODEL_PATH = MODELS / "model.pkl"
METRICS_PATH = MODELS / "metrics.json"

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "wine-quality/winequality-red.csv")
SEED = 42


# ---------------------------------------------------------------- étapes
def prepare():
    """Télécharge (ou lit en local), nettoie et sépare train/test."""
    DATA.mkdir(exist_ok=True)

    if RAW_CSV.exists():
        df = pd.read_csv(RAW_CSV, sep=";")
        print(f"[prepare] Lecture locale : {RAW_CSV}")
    else:
        print("[prepare] Téléchargement du dataset...")
        df = pd.read_csv(URL, sep=";")
        df.to_csv(RAW_CSV, sep=";", index=False)

    df = df.drop_duplicates().dropna()
    df["good"] = (df["quality"] >= 6).astype(int)  # bon vin si note >= 6
    df = df.drop(columns="quality")

    train, test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["good"]
    )
    train.to_csv(TRAIN_CSV, index=False)
    test.to_csv(TEST_CSV, index=False)
    print(f"[prepare] Train : {len(train)} lignes | Test : {len(test)} lignes")


def train():
    """Entraîne un RandomForest et sauvegarde le modèle."""
    MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(TRAIN_CSV)
    X, y = df.drop(columns="good"), df["good"]

    model = RandomForestClassifier(n_estimators=200, random_state=SEED)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"[train] Modèle sauvegardé : {MODEL_PATH}")


def evaluate():
    """Calcule accuracy et F1 sur le jeu de test."""
    df = pd.read_csv(TEST_CSV)
    X, y = df.drop(columns="good"), df["good"]

    model = joblib.load(MODEL_PATH)
    pred = model.predict(X)

    metrics = {
        "accuracy": round(accuracy_score(y, pred), 3),
        "f1": round(f1_score(y, pred), 3),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] {metrics}")
    return metrics


def test():
    """Tests automatiques simples (utilisables en CI)."""
    model = joblib.load(MODEL_PATH)
    X = pd.read_csv(TEST_CSV).drop(columns="good")
    pred = model.predict(X)

    assert len(pred) == len(X), "Nombre de prédictions incorrect"
    assert set(pred) <= {0, 1}, "Les prédictions doivent être 0 ou 1"

    metrics = json.loads(METRICS_PATH.read_text())
    assert metrics["accuracy"] >= 0.70, f"Accuracy trop faible : {metrics['accuracy']}"
    print("[test] Tous les tests sont OK")


