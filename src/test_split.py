import pandas as pd
from max_minority import trouver_meilleur_split

df = pd.read_csv("data/features.csv")
for col in ["pct_rouille", "rugosite", "nb_taches"]:
    seuil, p = trouver_meilleur_split(df[col].values, df["label_malade"].values)
    print(f"{col:15s} -> seuil={seuil}  P_split={p:.3f}")