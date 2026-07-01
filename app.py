"""
Partie 4 : Application Web Streamlit
Diagnostic de la Rouille Polysora — Feuilles de Maïs
Design : identite visuelle terrain, techniciens agricoles Madagascar
"""

import os
import sys
import pickle
from datetime import datetime

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from feature import compute_pct_rouille, compute_rugosite, compute_nb_taches

# ------------------------------------------------------------------ #
#  Config page                                                         #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="DiagMaïs · Rouille Polysora",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------ #
#  CSS personnalise                                                    #
# ------------------------------------------------------------------ #

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@400;500&display=swap');

/* Reset general */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header personnalise */
.diag-header {
    background: #1B3A2D;
    color: #F5F0E8;
    padding: 2rem 2.5rem 1.5rem 2.5rem;
    border-radius: 0 0 20px 20px;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.diag-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
    color: #F5F0E8;
}
.diag-header .subtitle {
    font-size: 0.9rem;
    color: #C8973A;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
.diag-header .icon {
    font-size: 3rem;
    line-height: 1;
}

/* Zone d'upload */
.upload-zone {
    border: 2.5px dashed #C8973A;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    background: #FAF7F2;
    margin-bottom: 1.5rem;
}

/* Carte de feature */
.feature-card {
    background: #1B3A2D;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #F5F0E8;
    margin-bottom: 0.8rem;
}
.feature-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #C8973A;
    font-weight: 600;
    margin-bottom: 0.2rem;
}
.feature-card .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
}

/* Verdict MALADE */
.verdict-malade {
    background: #C0392B;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.verdict-malade .tampon {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: #FF8A80;
    letter-spacing: 0.15em;
    display: block;
    transform: rotate(-3deg);
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.verdict-malade .detail {
    color: #FFD0CC;
    font-size: 0.95rem;
    margin-top: 0.8rem;
}
.verdict-malade .confiance {
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.verdict-malade .action {
    background: rgba(0,0,0,0.2);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: #FFD0CC;
    font-size: 0.85rem;
    margin-top: 1rem;
    text-align: left;
}

/* Verdict SAINE */
.verdict-saine {
    background: #2D7A4F;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.verdict-saine .tampon {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    color: #A8E6C3;
    letter-spacing: 0.15em;
    display: block;
    transform: rotate(-3deg);
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.verdict-saine .detail {
    color: #C8F0DC;
    font-size: 0.95rem;
    margin-top: 0.8rem;
}
.verdict-saine .confiance {
    color: white;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    margin-top: 0.5rem;
}
.verdict-saine .action {
    background: rgba(0,0,0,0.15);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: #C8F0DC;
    font-size: 0.85rem;
    margin-top: 1rem;
    text-align: left;
}

/* Titre de section */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1B3A2D;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-left: 4px solid #C8973A;
    padding-left: 0.8rem;
    margin: 2rem 0 1rem 0;
}

/* Carte historique */
.hist-badge-malade {
    background: #C0392B;
    color: white;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-align: center;
    margin-top: 4px;
}
.hist-badge-saine {
    background: #2D7A4F;
    color: white;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-align: center;
    margin-top: 4px;
}
.hist-date {
    font-size: 0.7rem;
    color: #888;
    text-align: center;
    margin-top: 2px;
}

/* Masquer le menu hamburger Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Constantes                                                          #
# ------------------------------------------------------------------ #

MODEL_PATH  = "models/meilleur_modele.pkl"
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
#  Chargement du modele                                               #
# ------------------------------------------------------------------ #

@st.cache_resource
def charger_modele():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["modele"]

# ------------------------------------------------------------------ #
#  Extraction des features                                             #
# ------------------------------------------------------------------ #

def extraire_features(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pct_rouille, mask = compute_pct_rouille(hsv)
    rugosite          = compute_rugosite(gray)
    nb_taches         = compute_nb_taches(mask)
    return np.array([[pct_rouille, rugosite, nb_taches]]), pct_rouille, rugosite, nb_taches

# ------------------------------------------------------------------ #
#  HEADER                                                              #
# ------------------------------------------------------------------ #

st.markdown("""
<div class="diag-header">
    <div class="icon">🌿</div>
    <div>
        <h1>DiagMaïs</h1>
        <div class="subtitle">Détection Rouille Polysora · Madagascar</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Chargement modele                                                   #
# ------------------------------------------------------------------ #

try:
    modele = charger_modele()
except FileNotFoundError:
    st.error("Modèle introuvable. Lance d'abord `python src/compare_models.py`.")
    st.stop()

# ------------------------------------------------------------------ #
#  MODULE 1 : Upload et Prediction                                     #
# ------------------------------------------------------------------ #

st.markdown('<div class="section-title">📤 Analyser une feuille</div>', unsafe_allow_html=True)

fichier = st.file_uploader(
    "Téléversez une photo de feuille de maïs (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if fichier is not None:
    img_bytes = fichier.read()

    col_img, col_features, col_verdict = st.columns([1.2, 0.9, 1.2])

    with col_img:
        st.image(img_bytes, caption="Image analysée", use_container_width=True)

    with col_features:
        st.markdown("**Caractéristiques extraites**")
        with st.spinner("Analyse en cours..."):
            try:
                features, pct_rouille, rugosite, nb_taches = extraire_features(img_bytes)
            except Exception as e:
                st.error(f"Erreur extraction : {e}")
                st.stop()

        st.markdown(f"""
        <div class="feature-card">
            <div class="label">Surface rouillée</div>
            <div class="value">{pct_rouille*100:.2f}<span style="font-size:1rem;font-weight:400"> %</span></div>
        </div>
        <div class="feature-card">
            <div class="label">Rugosité (Sobel)</div>
            <div class="value">{rugosite:.0f}</div>
        </div>
        <div class="feature-card">
            <div class="label">Taches détectées</div>
            <div class="value">{int(nb_taches)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_verdict:
        prediction = modele.predict(features)[0]
        proba      = modele.predict_proba(features)[0]

        if prediction == 1:
            st.markdown(f"""
            <div class="verdict-malade">
                <span class="tampon">MALADE</span>
                <div class="confiance">Confiance : {proba[1]*100:.1f}%</div>
                <div class="detail">Rouille Polysora (<i>Puccinia polysora</i>) détectée</div>
                <div class="action">
                    ⚠️ <strong>Action requise</strong><br>
                    Isoler la plante. Appliquer un fongicide adapté
                    (mancozèbe ou triazole). Surveiller les parcelles adjacentes.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-saine">
                <span class="tampon">SAINE</span>
                <div class="confiance">Confiance : {proba[0]*100:.1f}%</div>
                <div class="detail">Aucune trace de rouille détectée</div>
                <div class="action">
                    ✅ <strong>Situation normale</strong><br>
                    Surveillance régulière conseillée.
                    Prochain contrôle recommandé sous 7 jours.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Sauvegarde historique
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_str  = "malade" if prediction == 1 else "saine"
        ext        = os.path.splitext(fichier.name)[1].lower() or ".jpg"
        nom_fichier = f"{timestamp}_{label_str}{ext}"
        with open(os.path.join(UPLOADS_DIR, nom_fichier), "wb") as f:
            f.write(img_bytes)

# ------------------------------------------------------------------ #
#  MODULE 2 : Galerie historique                                       #
# ------------------------------------------------------------------ #

st.markdown('<div class="section-title">🗂️ Historique des analyses</div>', unsafe_allow_html=True)

images = sorted(
    [f for f in os.listdir(UPLOADS_DIR)
     if f.lower().endswith((".jpg", ".jpeg", ".png"))],
    reverse=True,
)

if not images:
    st.info("Aucune analyse effectuée pour le moment.")
else:
    nb_total  = len(images)
    nb_malades = sum(1 for f in images if "_malade" in f)
    nb_saines  = nb_total - nb_malades

    m1, m2, m3 = st.columns(3)
    m1.metric("Total analysées", nb_total)
    m2.metric("🔴 Malades", nb_malades)
    m3.metric("🟢 Saines", nb_saines)

    st.markdown("---")

    NB_COL = 5
    for i in range(0, len(images), NB_COL):
        cols = st.columns(NB_COL)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(images):
                break
            nom = images[idx]
            chemin = os.path.join(UPLOADS_DIR, nom)

            badge_html = (
                '<div class="hist-badge-malade">MALADE</div>'
                if "_malade" in nom
                else '<div class="hist-badge-saine">SAINE</div>'
            )

            parts = nom.split("_")
            try:
                ts = datetime.strptime(parts[0] + parts[1], "%Y%m%d%H%M%S")
                date_str = ts.strftime("%d/%m %H:%M")
            except (ValueError, IndexError):
                date_str = ""

            with col:
                st.image(chemin, use_container_width=True)
                st.markdown(badge_html, unsafe_allow_html=True)
                st.markdown(f'<div class="hist-date">{date_str}</div>', unsafe_allow_html=True)

    st.markdown("")
    if st.button("🗑️ Vider l'historique", type="secondary"):
        for f in images:
            os.remove(os.path.join(UPLOADS_DIR, f))
        st.rerun()