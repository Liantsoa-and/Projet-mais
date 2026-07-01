"""
Script de calibration interactive des seuils HSV pour détecter
les zones de rouille (Puccinia polysora) sur les feuilles de maïs.

Usage :
    python calibrate_hsv.py dataset/malades/<nom_image>.jpg

Les valeurs actuelles des 6 seuils sont affichées EN DIRECT dans le
terminal (pas besoin de lire les labels sur les sliders, qui peuvent
être tronqués selon l'environnement Linux/Qt).

Ajuste les trackbars jusqu'à ce que la fenêtre "Masque" isole
proprement les pustules orangées/brunâtres (blanc = rouille détectée,
noir = reste de la feuille / fond).

Appuie sur 'q' (fenêtre image active) pour quitter et figer les valeurs.
"""

import sys
import cv2
import numpy as np


def nothing(x):
    pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate_hsv.py <chemin_image>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Erreur : impossible de charger l'image {img_path}")
        sys.exit(1)

    max_dim = 600
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.namedWindow("Controles", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controles", 500, 350)  # fenetre large pour voir les labels

    cv2.createTrackbar("H_min", "Controles", 5, 179, nothing)
    cv2.createTrackbar("H_max", "Controles", 35, 179, nothing)
    cv2.createTrackbar("S_min", "Controles", 50, 255, nothing)
    cv2.createTrackbar("S_max", "Controles", 255, 255, nothing)
    cv2.createTrackbar("V_min", "Controles", 50, 255, nothing)
    cv2.createTrackbar("V_max", "Controles", 255, 255, nothing)

    print("Ajuste les trackbars (les valeurs s'affichent ci-dessous en direct).")
    print("Appuie sur 'q' (fenetre image active) pour valider et quitter.\n")

    last_printed = None

    while True:
        h_min = cv2.getTrackbarPos("H_min", "Controles")
        h_max = cv2.getTrackbarPos("H_max", "Controles")
        s_min = cv2.getTrackbarPos("S_min", "Controles")
        s_max = cv2.getTrackbarPos("S_max", "Controles")
        v_min = cv2.getTrackbarPos("V_min", "Controles")
        v_max = cv2.getTrackbarPos("V_max", "Controles")

        current = (h_min, h_max, s_min, s_max, v_min, v_max)
        if current != last_printed:
            print(
                f"H_min={h_min:3d}  H_max={h_max:3d}  "
                f"S_min={s_min:3d}  S_max={s_max:3d}  "
                f"V_min={v_min:3d}  V_max={v_max:3d}",
                end="\r",
            )
            last_printed = current

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("Image originale", img)
        cv2.imshow("Masque", mask)
        cv2.imshow("Zones detectees", result)

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            print("\n\nValeurs retenues :")
            print(f"  lower = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"  upper = np.array([{h_max}, {s_max}, {v_max}])")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()