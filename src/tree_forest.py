"""
Partie 3 : Arbre de Decision et Random Forest "From Scratch"
          Metrique Max-Minority

Contenu :
  - build_tree()          : arbre de decision recursif
  - predict_one/predict() : prediction sur l'arbre
  - afficher_arbre()      : affichage lisible de la structure
  - build_random_forest() : foret aleatoire par bagging
  - predict_forest()      : vote majoritaire sur la foret
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from max_minority import purete, trouver_meilleur_split


# ------------------------------------------------------------------ #
#  Utilitaire                                                          #
# ------------------------------------------------------------------ #

def classe_majoritaire(y):
    """Retourne la classe la plus frequente dans y (0 ou 1)."""
    n_1 = np.sum(y == 1)
    n_0 = len(y) - n_1
    return 1 if n_1 >= n_0 else 0


# ------------------------------------------------------------------ #
#  Arbre de Decision From Scratch                                      #
# ------------------------------------------------------------------ #

def build_tree(X, y, feature_names, depth=0, max_depth=3):
    """
    Construit recursivement un arbre de decision Max-Minority.

    X             : np.ndarray (n_samples, n_features)
    y             : np.ndarray de labels (0 ou 1)
    feature_names : liste des noms de colonnes
    depth         : profondeur courante (ne pas passer manuellement)
    max_depth     : profondeur maximale autorisee

    Retourne un dictionnaire representant le noeud courant.
    """
    # Conditions d'arret -> feuille
    if depth >= max_depth:
        return {"feuille": True, "classe": classe_majoritaire(y)}
    if purete(y) == 1.0:
        return {"feuille": True, "classe": classe_majoritaire(y)}
    if len(y) <= 1:
        return {"feuille": True, "classe": classe_majoritaire(y)}

    # Chercher le meilleur split sur toutes les variables
    meilleur_feature = None
    meilleur_seuil = None
    meilleure_purete = -np.inf
    meilleur_feature_idx = None

    for i, feature in enumerate(feature_names):
        seuil, p_split = trouver_meilleur_split(X[:, i], y)
        if seuil is not None and p_split > meilleure_purete:
            meilleure_purete = p_split
            meilleur_seuil = seuil
            meilleur_feature = feature
            meilleur_feature_idx = i

    if meilleur_feature is None:
        return {"feuille": True, "classe": classe_majoritaire(y)}

    # Diviser les donnees
    masque_gauche = X[:, meilleur_feature_idx] <= meilleur_seuil
    masque_droite = ~masque_gauche
    X_gauche, y_gauche = X[masque_gauche], y[masque_gauche]
    X_droite, y_droite = X[masque_droite], y[masque_droite]

    if len(y_gauche) == 0 or len(y_droite) == 0:
        return {"feuille": True, "classe": classe_majoritaire(y)}

    # Appel recursif
    return {
        "feuille": False,
        "variable": meilleur_feature,
        "variable_idx": meilleur_feature_idx,
        "seuil": meilleur_seuil,
        "gauche": build_tree(X_gauche, y_gauche, feature_names, depth + 1, max_depth),
        "droite": build_tree(X_droite, y_droite, feature_names, depth + 1, max_depth),
    }


def predict_one(noeud, x):
    """Predit la classe d'UN seul exemple x en parcourant l'arbre."""
    if noeud["feuille"]:
        return noeud["classe"]
    if x[noeud["variable_idx"]] <= noeud["seuil"]:
        return predict_one(noeud["gauche"], x)
    else:
        return predict_one(noeud["droite"], x)


def predict(noeud, X):
    """Predit la classe de TOUS les exemples de X."""
    return np.array([predict_one(noeud, x) for x in X])


def afficher_arbre(noeud, indent=0):
    """Affiche la structure de l'arbre dans le terminal."""
    prefix = "  " * indent
    if noeud["feuille"]:
        label = "MALADE" if noeud["classe"] == 1 else "SAINE"
        print(f"{prefix}-> FEUILLE : predire {label} ({noeud['classe']})")
    else:
        print(f"{prefix}[{noeud['variable']} <= {noeud['seuil']:.4f}]")
        print(f"{prefix}  Gauche (oui) :")
        afficher_arbre(noeud["gauche"], indent + 2)
        print(f"{prefix}  Droite (non) :")
        afficher_arbre(noeud["droite"], indent + 2)


# ------------------------------------------------------------------ #
#  Random Forest From Scratch (Bagging + Vote Majoritaire)            #
# ------------------------------------------------------------------ #

def build_random_forest(X, y, feature_names, n_trees=10, max_depth=3):
    """
    Construit une foret aleatoire composee de n_trees arbres Max-Minority.

    Principe du Bagging : chaque arbre est entraine sur un sous-echantillon
    tire AVEC REMPLACEMENT depuis X (bootstrap). Certaines lignes
    apparaissent plusieurs fois, d'autres pas du tout -> chaque arbre
    voit une vision legerement differente des donnees -> diversite.

    X             : np.ndarray (n_samples, n_features)
    y             : np.ndarray de labels (0 ou 1)
    feature_names : liste des noms de colonnes
    n_trees       : nombre d'arbres dans la foret
    max_depth     : profondeur max de chaque arbre

    Retourne une liste d'arbres (un dictionnaire par arbre).
    """
    foret = []
    n_samples = len(y)

    for i in range(n_trees):
        # Tirage bootstrap : n_samples indices avec remplacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X[indices]
        y_bootstrap = y[indices]

        arbre = build_tree(X_bootstrap, y_bootstrap, feature_names, max_depth=max_depth)
        foret.append(arbre)
        print(f"  Arbre {i+1}/{n_trees} construit.")

    return foret


def predict_forest(foret, X):
    """
    Predit la classe de tous les exemples par vote majoritaire.

    Pour chaque exemple, chaque arbre vote -> la classe qui obtient
    le plus de votes gagne.

    foret : liste d'arbres retournee par build_random_forest
    X     : np.ndarray (n_samples, n_features)

    Retourne un np.ndarray de predictions (0 ou 1).
    """
    # shape : (n_trees, n_samples)
    toutes_predictions = np.array([predict(arbre, X) for arbre in foret])

    # Vote majoritaire par colonne (= par exemple)
    predictions_finales = []
    for j in range(X.shape[0]):
        votes = toutes_predictions[:, j]
        predictions_finales.append(1 if np.sum(votes == 1) >= np.sum(votes == 0) else 0)

    return np.array(predictions_finales)


# ------------------------------------------------------------------ #
#  Test complet : Arbre + Random Forest From Scratch                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/features.csv")
    feature_names = ["pct_rouille", "rugosite", "nb_taches"]

    X = df[feature_names].values
    y = df["label_malade"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train : {len(X_train)} exemples  |  Test : {len(X_test)} exemples")

    # --- Arbre seul ---
    print(f"\n{'='*50}")
    print("Arbre de Decision From Scratch (max_depth=3)")
    print(f"{'='*50}")
    arbre = build_tree(X_train, y_train, feature_names, max_depth=3)
    afficher_arbre(arbre)
    y_pred_arbre = predict(arbre, X_test)
    accuracy_arbre = np.mean(y_pred_arbre == y_test)
    print(f"\nAccuracy Arbre : {accuracy_arbre:.2f}")

    # --- Random Forest ---
    print(f"\n{'='*50}")
    print("Random Forest From Scratch (10 arbres, max_depth=3)")
    print(f"{'='*50}")
    np.random.seed(42)
    foret = build_random_forest(X_train, y_train, feature_names, n_trees=10, max_depth=3)
    y_pred_rf = predict_forest(foret, X_test)
    accuracy_rf = np.mean(y_pred_rf == y_test)
    print(f"\nAccuracy Random Forest : {accuracy_rf:.2f}")

    # --- Comparaison rapide ---
    print(f"\n{'='*50}")
    print("Comparaison rapide")
    print(f"{'='*50}")
    print(f"  Arbre From Scratch     : {accuracy_arbre:.2f}")
    print(f"  Random Forest Scratch  : {accuracy_rf:.2f}")
    print(f"\n(Les metriques completes Precision/Rappel seront dans compare_models.py)")