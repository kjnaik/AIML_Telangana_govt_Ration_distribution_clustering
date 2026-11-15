from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from features import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def prepare(shops: pd.DataFrame):
    """Return modelling frame, clipped+scaled matrix (K-Means) and raw-scaled matrix (DBSCAN)."""
    m = shops[shops["use_for_model"]].copy()
    X = m[MODEL_FEATURES].astype(float)

    # Version for K-Means/PCA: clip extremes so they don't dominate
    lo, hi = X.quantile(0.005), X.quantile(0.995)
    X_clip = X.clip(lo, hi, axis=1)
    scaler = StandardScaler().fit(X_clip)
    Z = scaler.transform(X_clip)

    # Version for DBSCAN: no clipping so extreme shops stay extreme
    scaler_raw = StandardScaler().fit(X)
    Z_raw = scaler_raw.transform(X)
    return m, Z, Z_raw, scaler, (lo, hi)


def k_search(Z, k_range=range(2, 9), sample=6000):
    rng = np.random.RandomState(42)
    idx = rng.choice(len(Z), size=min(sample, len(Z)), replace=False)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
        sil = silhouette_score(Z[idx], km.labels_[idx])
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def run_dbscan(Z_raw, min_samples=20):
    # k-distance heuristic for eps: 98th percentile of distance to the k-th neighbour
    nn = NearestNeighbors(n_neighbors=min_samples).fit(Z_raw)
    d, _ = nn.kneighbors(Z_raw)
    eps = float(np.percentile(d[:, -1], 98))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z_raw)
    return db.labels_, eps


if __name__ == "__main__":
    shops = pd.read_csv(PROCESSED / "shop_features.csv")
    m, Z, Z_raw, scaler, bounds = prepare(shops)
    print("Modelling shops:", len(m), "| features:", MODEL_FEATURES)

    ks = k_search(Z)
    print("\nElbow / silhouette:\n", ks.round(3))
    ks.to_csv(PROCESSED / "k_search.csv", index=False)
    print("\nBest k by silhouette:", int(ks.loc[ks.silhouette.idxmax(), "k"]))