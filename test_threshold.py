"""
Vérifie les seuils HSV fusionnés sur TOUTES les images du dataset
(saines + malades) et sauvegarde les masques générés dans un dossier
pour inspection visuelle rapide.

Usage :
    python test_threshold.py

Résultat :
    Crée un dossier calibration_check/ contenant, pour chaque image,
    un montage [Original | Masque | Zones detectees].
    Objectif visuel :
      - dataset/malades/*  -> le masque doit capturer les pustules
      - dataset/saines/*   -> le masque doit rester quasi tout noir
"""

import os
import cv2
import numpy as np

# --- Seuils a tester (modifie ici si besoin) ---
LOWER = np.array([5, 59, 209])
UPPER = np.array([37, 255, 255])

DATASET_DIRS = {
    "malades": "dataset/malades",
    "saines": "dataset/saines",
}

OUTPUT_DIR = "calibration_check"


def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None

    max_dim = 500
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER, UPPER)
    result = cv2.bitwise_and(img, img, mask=mask)

    pct_rouille = (mask > 0).sum() / mask.size

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    montage = np.hstack([img, mask_bgr, result])

    return montage, pct_rouille


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Seuils testes : lower={LOWER.tolist()}  upper={UPPER.tolist()}\n")

    for category, folder in DATASET_DIRS.items():
        if not os.path.isdir(folder):
            print(f"Dossier introuvable : {folder}")
            continue

        for filename in sorted(os.listdir(folder)):
            img_path = os.path.join(folder, filename)
            res = process_image(img_path)
            if res is None:
                continue
            montage, pct_rouille = res

            out_name = f"{category}_{filename}"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, montage)

            print(f"[{category}] {filename:30s} pct_rouille = {pct_rouille:.3f}  -> {out_path}")

    print(f"\nTermine. Ouvre le dossier '{OUTPUT_DIR}/' pour inspecter les montages.")
    print("Attendu : pct_rouille proche de 0 pour les saines, nettement > 0 pour les malades.")


if __name__ == "__main__":
    main()