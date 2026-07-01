"""
Partie 2 : Le Defi Algorithmique - L'Indice "Max-Minority"

Implemente la recherche du meilleur seuil de split pour une variable
continue, en utilisant la metrique de purete Max-Minority :

    P(t) = max_c (n_c / N)

et la purete ponderee d'un split :

    P_split = (|G|/N) * P(G) + (|D|/N) * P(D)

But : trouver le seuil qui MAXIMISE P_split (contrairement a Gini ou
on minimise l'impurete, ici on maximise directement la purete).
"""

import numpy as np


def purete(y):
    """
    Calcule P(t) = proportion de la classe majoritaire dans y.

    y : array-like de labels (0 ou 1)
    Retourne un flottant entre 0.5 (melange parfait) et 1.0 (noeud pur).
    Si y est vide, retourne 0 (cas degenere, ne devrait pas arriver
    si le code appelant verifie bien que G et D sont non vides).
    """
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        return 0.0

    n_classe_1 = np.sum(y == 1)
    n_classe_0 = n - n_classe_1

    return max(n_classe_0, n_classe_1) / n


def trouver_meilleur_split(X_column, y):
    """
    Teste tous les seuils candidats pour une variable continue et
    retourne celui qui maximise la purete ponderee du split (P_split).

    X_column : array-like des valeurs de la variable a tester
               (ex : la colonne pct_rouille)
    y        : array-like des labels correspondants (0 ou 1)

    Retourne :
        (meilleur_seuil, meilleure_purete)
        meilleur_seuil  : la valeur de seuil s optimale (float)
        meilleure_purete: la valeur de P_split associee (float)
        Si aucun split valide n'existe (ex: une seule valeur unique
        dans X_column), retourne (None, purete(y)) -- impossible de
        splitter, donc on renvoie la purete du noeud parent tel quel.
    """
    X_column = np.asarray(X_column)
    y = np.asarray(y)
    N = len(y)

    # etape 1 : trier les donnees par X_column
    ordre_tri = np.argsort(X_column)
    X_trie = X_column[ordre_tri]
    y_trie = y[ordre_tri]

    # valeurs uniques triees -> necessaires pour generer les seuils candidats
    valeurs_uniques = np.unique(X_trie)

    # etape 2 : initialiser le suivi du meilleur seuil / meilleure purete
    meilleur_seuil = None
    meilleure_purete = -np.inf

    # s'il n'y a qu'une seule valeur unique, aucun split n'est possible
    if len(valeurs_uniques) < 2:
        return None, purete(y_trie)

    # etape 3 : parcourir les seuils candidats (milieu entre 2 valeurs uniques consecutives)
    for i in range(len(valeurs_uniques) - 1):
        seuil_candidat = (valeurs_uniques[i] + valeurs_uniques[i + 1]) / 2.0

        # separation virtuelle : Gauche = X <= seuil, Droite = X > seuil
        masque_gauche = X_trie <= seuil_candidat
        masque_droite = ~masque_gauche

        y_gauche = y_trie[masque_gauche]
        y_droite = y_trie[masque_droite]

        n_gauche = len(y_gauche)
        n_droite = len(y_droite)

        # purete ponderee du split (formule du sujet)
        p_split = (n_gauche / N) * purete(y_gauche) + (n_droite / N) * purete(y_droite)

        if p_split > meilleure_purete:
            meilleure_purete = p_split
            meilleur_seuil = seuil_candidat

    # etape 4 : retourner le seuil optimal et sa purete associee
    return meilleur_seuil, meilleure_purete


if __name__ == "__main__":
    # --- Petit test manuel avec l'exemple du cours pour verifier la logique ---
    X_test = np.array([0.02, 0.05, 0.18, 0.22, 0.31])
    y_test = np.array([0, 0, 1, 1, 1])

    seuil, p = trouver_meilleur_split(X_test, y_test)
    print(f"Test manuel : meilleur_seuil={seuil:.3f}  P_split={p:.3f}")
    # Attendu (calcule a la main) : seuil=0.20, P_split=0.8 (cf. explication donnee)