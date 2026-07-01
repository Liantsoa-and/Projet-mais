"""
Partie 1 : Feature Engineering - Du Pixel aux Caracteristiques

Parcourt dataset/saines/ et dataset/malades/, extrait pour chaque image :
  - pct_rouille : proportion de pixels "rouille" (masque HSV calibre)
  - rugosite    : variance des gradients de Sobel (texture)
  - nb_taches   : nombre de composantes connexes dans le masque rouille
                  (justification agronomique : la rouille se manifeste par
                  de multiples pustules dispersees, alors qu'une grande
                  zone connexe unique evoque plutot un artefact/une autre
                  pathologie ou une ombre mal filtree)

Sauvegarde le resultat dans data/features.csv avec les colonnes :
    [ID_Image | pct_rouille | rugosite | nb_taches | label_malade]

Usage :
    python src/features.py
"""

import os
import cv2
import numpy as np
import pandas as pd

# --- Seuils HSV calibres manuellement (cf. calibrate_hsv.py / test_threshold.py) ---
LOWER_ROUILLE = np.array([5, 59, 209])
UPPER_ROUILLE = np.array([37, 255, 255])

DATASET_DIRS = {
    "saines": ("dataset/saines", 0),
    "malades": ("dataset/malades", 1),
}

OUTPUT_CSV = "data/features.csv"

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def compute_pct_rouille(hsv_img):
    """X1 : proportion de pixels de rouille / pixels totaux de l'image."""
    mask = cv2.inRange(hsv_img, LOWER_ROUILLE, UPPER_ROUILLE)
    pct = (mask > 0).sum() / mask.size
    return pct, mask


def compute_rugosite(gray_img):
    """X2 : variance de la magnitude des gradients de Sobel (texture/rugosite)."""
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return float(np.var(magnitude))


def compute_nb_taches(mask, min_area=5):
    """
    X3 (variable personnelle) : nombre de composantes connexes du masque
    rouille, en ignorant les taches trop petites (bruit/pixels isoles).
    """
    n_labels, labels = cv2.connectedComponents(mask)
    nb_taches = 0
    for label_id in range(1, n_labels):  # 0 = fond, on l'ignore
        area = (labels == label_id).sum()
        if area >= min_area:
            nb_taches += 1
    return nb_taches


def extract_features_from_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [!] Image illisible, ignoree : {img_path}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pct_rouille, mask = compute_pct_rouille(hsv)
    rugosite = compute_rugosite(gray)
    nb_taches = compute_nb_taches(mask)

    return {
        "pct_rouille": pct_rouille,
        "rugosite": rugosite,
        "nb_taches": nb_taches,
    }


def main():
    rows = []

    for category, (folder, label) in DATASET_DIRS.items():
        if not os.path.isdir(folder):
            print(f"Dossier introuvable, ignore : {folder}")
            continue

        print(f"Traitement de {folder} (label={label}) ...")
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith(IMG_EXTENSIONS):
                continue

            img_path = os.path.join(folder, filename)
            features = extract_features_from_image(img_path)
            if features is None:
                continue

            row = {
                "ID_Image": filename,
                "pct_rouille": features["pct_rouille"],
                "rugosite": features["rugosite"],
                "nb_taches": features["nb_taches"],
                "label_malade": label,
            }
            rows.append(row)
            print(
                f"  {filename:30s} pct_rouille={features['pct_rouille']:.3f}  "
                f"rugosite={features['rugosite']:.1f}  nb_taches={features['nb_taches']}"
            )

    df = pd.DataFrame(rows, columns=["ID_Image", "pct_rouille", "rugosite", "nb_taches", "label_malade"])

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{len(df)} images traitees. Resultat sauvegarde dans {OUTPUT_CSV}")
    print("\nApercu :")
    print(df)


if __name__ == "__main__":
    main()