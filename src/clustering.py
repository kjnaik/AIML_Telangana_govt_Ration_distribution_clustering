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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = 4
    FIG = ROOT / "reports" / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    shops = pd.read_csv(PROCESSED / "shop_features.csv")
    m, Z, Z_raw, scaler, bounds = prepare(shops)

    # --- Elbow + silhouette figure ---
    ks = pd.read_csv(PROCESSED / "k_search.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ks.k, ks.inertia, marker="o"); ax[0].set_title("Elbow curve (inertia)")
    ax[1].plot(ks.k, ks.silhouette, marker="o", color="#e76f51")
    ax[1].axvline(K, ls="--", color="grey"); ax[1].set_title(f"Silhouette score (chosen k={K})")
    for a in ax: a.set_xlabel("k")
    plt.tight_layout(); plt.savefig(FIG / "elbow_silhouette.png", dpi=150); plt.close()

    # --- Final K-Means ---
    km = KMeans(n_clusters=K, n_init=20, random_state=42).fit(Z)
    m["cluster"] = km.labels_

    # --- PCA ---
    pca = PCA(n_components=3, random_state=42).fit(Z)
    m[["pc1", "pc2", "pc3"]] = pca.transform(Z)
    print("PCA explained variance:", pca.explained_variance_ratio_.round(3),
          "| cumulative (2D):", round(pca.explained_variance_ratio_[:2].sum(), 3))
    loadings = pd.DataFrame(pca.components_[:3].T, index=MODEL_FEATURES,
                            columns=["PC1", "PC2", "PC3"]).round(2)
    print("\nPCA loadings:\n", loadings)

    # --- DBSCAN (unclipped data, so extreme shops stay extreme) ---
    labels, eps = run_dbscan(Z_raw)
    m["dbscan_label"] = labels
    m["is_outlier"] = labels == -1
    print(f"\nDBSCAN eps={eps:.3f} | noise shops: {m.is_outlier.sum()} "
          f"({m.is_outlier.mean():.1%}) | dense clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    print("\nK-Means cluster vs DBSCAN outlier:\n", pd.crosstab(m.cluster, m.is_outlier))

    # --- Cluster profile (mean of every feature) ---
    cols = MODEL_FEATURES + ["avg_trans", "avg_cards", "util", "local_util"]
    profile = m.groupby("cluster")[cols].mean()
    profile["shops"] = m.groupby("cluster").size()
    profile["pct_outlier"] = m.groupby("cluster")["is_outlier"].mean() * 100
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("\nCLUSTER PROFILE:\n", profile.round(3).T)
    profile.round(4).to_csv(PROCESSED / "cluster_profile.csv")

    # --- Urban vs rural mix (for the cluster-purity metric) ---
    urban_kw = ["hyderabad", "medchal", "ranga"]
    m["urban_zone"] = m["distName"].str.lower().str.contains("|".join(urban_kw))
    print("\nDistricts treated as urban:", sorted(m.loc[m.urban_zone, "distName"].unique()))
    print("Share of each cluster in urban districts:\n",
          m.groupby("cluster")["urban_zone"].mean().round(3))

    # --- PCA scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(m.pc1, m.pc2, c=m.cluster, cmap="tab10", s=4, alpha=.5)
    ax.scatter(m.loc[m.is_outlier, "pc1"], m.loc[m.is_outlier, "pc2"],
               facecolors="none", edgecolors="red", s=25, lw=.6, label="DBSCAN noise")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(*sc.legend_elements(), title="Cluster", loc="upper right")
    ax.set_title("Shops in PCA space (K-Means colours, red rings = DBSCAN noise)")
    plt.tight_layout(); plt.savefig(FIG / "pca_clusters.png", dpi=150); plt.close()

    # --- Save models and results ---
    joblib.dump({"scaler": scaler, "pca": pca, "kmeans": km, "clip_bounds": bounds,
                 "features": MODEL_FEATURES}, MODELS / "pds_models.pkl")
    m.to_csv(PROCESSED / "shop_clusters.csv", index=False)
    print("\nSaved: shop_clusters.csv, cluster_profile.csv, models/pds_models.pkl, figures")