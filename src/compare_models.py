"""
Partie 3 : Comparaison des 4 modeles

1. Arbre From Scratch      (Max-Minority)
2. Random Forest Scratch   (Max-Minority)
3. DecisionTreeClassifier  (scikit-learn, Gini)
4. RandomForestClassifier  (scikit-learn, Gini)

Metriques calculees sur le jeu de test (20%) :
  - Accuracy  = (VP + VN) / Total
  - Precision = VP / (VP + FP)
  - Rappel    = VP / (VP + FN)
  - Matrice de confusion

Usage :
    python src/compare_models.py
"""

import numpy as np
import pandas as pd
import sys
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from tree_forest import build_tree, predict, build_random_forest, predict_forest


# ------------------------------------------------------------------ #
#  Metriques manuelles                                                 #
# ------------------------------------------------------------------ #

def calculer_metriques(y_test, y_pred, nom_modele):
    """
    Calcule Accuracy, Precision, Rappel et la matrice de confusion.

    Convention : classe POSITIVE = 1 (MALADE)
                 classe NEGATIVE = 0 (SAINE)

    VP (Vrai Positif)  : predit MALADE, vraiment MALADE
    VN (Vrai Negatif)  : predit SAINE,  vraiment SAINE
    FP (Faux Positif)  : predit MALADE, vraiment SAINE  -> gaspillage traitement
    FN (Faux Negatif)  : predit SAINE,  vraiment MALADE -> epidemie non detectee
    """
    VP = np.sum((y_pred == 1) & (y_test == 1))
    VN = np.sum((y_pred == 0) & (y_test == 0))
    FP = np.sum((y_pred == 1) & (y_test == 0))
    FN = np.sum((y_pred == 0) & (y_test == 1))

    accuracy  = (VP + VN) / len(y_test)
    precision = VP / (VP + FP) if (VP + FP) > 0 else 0.0
    rappel    = VP / (VP + FN) if (VP + FN) > 0 else 0.0

    return {
        "Modele"   : nom_modele,
        "Accuracy" : round(accuracy, 3),
        "Precision": round(precision, 3),
        "Rappel"   : round(rappel, 3),
        "VP": VP, "VN": VN, "FP": FP, "FN": FN,
    }


def afficher_matrice_confusion(res):
    """Affiche la matrice de confusion 2x2 dans le terminal."""
    print(f"\n  Matrice de confusion — {res['Modele']}")
    print(f"  {'':20s} | Predit SAINE | Predit MALADE")
    print(f"  {'-'*52}")
    print(f"  {'Reel SAINE':20s} |     VN={res['VN']:3d}    |     FP={res['FP']:3d}")
    print(f"  {'Reel MALADE':20s} |     FN={res['FN']:3d}    |     VP={res['VP']:3d}")


# ------------------------------------------------------------------ #
#  Programme principal                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":

    # --- Chargement des donnees ---
    df = pd.read_csv("data/features.csv")
    feature_names = ["pct_rouille", "rugosite", "nb_taches"]

    X = df[feature_names].values
    y = df["label_malade"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Dataset : {len(df)} images  |  Train : {len(X_train)}  |  Test : {len(X_test)}")
    print(f"Distribution test -> Saines : {np.sum(y_test==0)}  Malades : {np.sum(y_test==1)}\n")

    resultats = []

    # ----------------------------------------------------------------
    # Modele 1 : Arbre From Scratch (Max-Minority)
    # ----------------------------------------------------------------
    print("[ 1/4 ] Construction Arbre From Scratch ...")
    arbre_scratch = build_tree(X_train, y_train, feature_names, max_depth=3)
    y_pred_1 = predict(arbre_scratch, X_test)
    resultats.append(calculer_metriques(y_test, y_pred_1, "Arbre Scratch (Max-Min)"))
    print("        OK")

    # ----------------------------------------------------------------
    # Modele 2 : Random Forest From Scratch (Max-Minority)
    # ----------------------------------------------------------------
    print("[ 2/4 ] Construction Random Forest From Scratch ...")
    np.random.seed(42)
    foret_scratch = build_random_forest(
        X_train, y_train, feature_names, n_trees=10, max_depth=3
    )
    y_pred_2 = predict_forest(foret_scratch, X_test)
    resultats.append(calculer_metriques(y_test, y_pred_2, "RF Scratch (Max-Min)"))
    print("        OK")

    # ----------------------------------------------------------------
    # Modele 3 : DecisionTreeClassifier scikit-learn (Gini)
    # ----------------------------------------------------------------
    print("[ 3/4 ] Entrainement DecisionTreeClassifier scikit-learn ...")
    clf_tree = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
    clf_tree.fit(X_train, y_train)
    y_pred_3 = clf_tree.predict(X_test)
    resultats.append(calculer_metriques(y_test, y_pred_3, "Arbre sklearn (Gini)"))
    print("        OK")

    # ----------------------------------------------------------------
    # Modele 4 : RandomForestClassifier scikit-learn (Gini)
    # ----------------------------------------------------------------
    print("[ 4/4 ] Entrainement RandomForestClassifier scikit-learn ...")
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_train)
    y_pred_4 = clf_rf.predict(X_test)
    resultats.append(calculer_metriques(y_test, y_pred_4, "RF sklearn (Gini)"))
    print("        OK\n")

    # ----------------------------------------------------------------
    # Tableau comparatif
    # ----------------------------------------------------------------
    print("=" * 65)
    print("TABLEAU COMPARATIF DES 4 MODELES")
    print("=" * 65)
    df_res = pd.DataFrame(resultats)[["Modele", "Accuracy", "Precision", "Rappel"]]
    print(df_res.to_string(index=False))
    print("=" * 65)

    # ----------------------------------------------------------------
    # Matrices de confusion detaillees
    # ----------------------------------------------------------------
    print("\nDETAIL DES MATRICES DE CONFUSION")
    for res in resultats:
        afficher_matrice_confusion(res)

    # ----------------------------------------------------------------
    # Sauvegarde du meilleur modele (Random Forest sklearn)
    # ----------------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    with open("models/meilleur_modele.pkl", "wb") as f:
        pickle.dump({
            "modele": clf_rf,
            "feature_names": feature_names
        }, f)
    print("\nMeilleur modele sauvegarde dans models/meilleur_modele.pkl")
    print("(utilise par app.py en Partie 4)")