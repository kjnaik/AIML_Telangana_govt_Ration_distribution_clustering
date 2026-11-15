# Telangana PDS Analytics: Shop Performance Clustering & Anomaly Profiling

Segments ~17,250 Fair Price Shops (FPS) in Telangana into behavioural personas using
transaction, card-status and location data, flags anomalous shops with DBSCAN, and serves
the results in an interactive Streamlit dashboard.

## Data
Telangana Open Data Portal (transactions, card status, FPS locations).
Months used: Jan 2023, Nov 2024, Feb 2025, Apr 2025, May 2025.
Raw CSVs are not committed (`data/raw/` is git-ignored).

## Method
1. **Consolidation** (`src/data_loader.py`): loop over monthly CSVs, join transactions + card
   status on `distCode, shopNo, month, year`, then add locations on `distCode, shopNo`.
2. **EDA** (`notebooks/01_eda.ipynb`): portability trend, entitlement vs transactions,
   commodity distributions, seasonality.
3. **Features** (`src/features.py`): one row per shop. Final clustering features:
   `log_avg_trans`, `rice_per_card`, `port_share`, `wheat_flag`, `trans_cv`, `port_growth`.
   Redundant features (correlation > 0.85) were removed.
4. **Clustering** (`src/clustering.py`): clip at 0.5/99.5 percentiles, StandardScaler, PCA,
   K-Means (k=4, chosen from elbow and silhouette), DBSCAN on unclipped scaled data.
5. **Dashboard** (`app.py`): filters, Folium map, cluster profiles, shop search.

## Personas
| Cluster | Persona | Shops |
|---|---|---|
| 0 | Urban Portability Hubs | 1,338 |
| 1 | Typical Rural Local Shops | 10,931 |
| 2 | Volatile Low-Volume Shops | 1,018 |
| 3 | Semi-Urban Portability Shops | 3,964 |

DBSCAN flagged 152 outlier shops (0.9%); 117 of them fall in cluster 2.

## Run it
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python src/data_loader.py      # needs raw CSVs in data/raw/
python src/features.py
python src/clustering.py
python src/make_dashboard_data.py
streamlit run app.py
```
The dashboard only needs the committed files in `data/processed/`.

## Structure
```
app.py            Streamlit dashboard
src/              data_loader, features, clustering, make_dashboard_data
notebooks/        01_eda.ipynb
data/processed/   shop_features, shop_clusters, cluster_profile, shop_year
models/           pds_models.pkl (scaler, PCA, K-Means)
reports/          final report and figures
```

## Limitations
Only five months of data, so seasonality cannot be confirmed. Urban/rural purity uses a
district-name approximation. Flagged shops need field verification and are not proof of fraud.