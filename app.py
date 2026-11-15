from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Telangana PDS Analytics", layout="wide")

DATA = Path(__file__).parent / "data" / "processed"
KEYS = ["distCode", "shopNo"]
PERSONA_COLORS = {
    "Urban Portability Hubs": "#d62728",
    "Typical Rural Local Shops": "#2ca02c",
    "Volatile Low-Volume Shops": "#ff7f0e",
    "Semi-Urban Portability Shops": "#1f77b4",
}
COMPARE = {
    "avg_trans": "Avg monthly transactions",
    "avg_cards": "Avg entitled cards",
    "util": "Utilization (trans / cards)",
    "port_share": "Portability share",
    "rice_per_card": "Rice per card (kg)",
    "trans_cv": "Volatility (CV)",
    "port_growth": "Portability growth",
}


@st.cache_data
def load():
    shops = pd.read_csv(DATA / "shop_clusters.csv")
    years = pd.read_csv(DATA / "shop_year.csv")
    prof = pd.read_csv(DATA / "cluster_profile.csv")
    return shops, years, prof


shops, years, prof = load()

# ---------------- Sidebar filters ----------------
st.sidebar.title("Filters")
districts = sorted(shops["distName"].dropna().unique())
sel_dist = st.sidebar.multiselect("District", districts, default=districts)
personas = list(PERSONA_COLORS)
sel_pers = st.sidebar.multiselect("Persona", personas, default=personas)
year_opts = ["All years"] + sorted(years["year"].unique().tolist())
sel_year = st.sidebar.selectbox("Year", year_opts)
only_out = st.sidebar.checkbox("DBSCAN outliers only")

f = shops[shops["distName"].isin(sel_dist) & shops["persona"].isin(sel_pers)].copy()
if only_out:
    f = f[f["is_outlier"]]
if sel_year != "All years":
    y = years[years["year"] == sel_year][KEYS + ["avg_trans", "port_share", "util"]]
    f = f.drop(columns=["avg_trans", "port_share", "util"]).merge(y, on=KEYS, how="inner")

st.title("Telangana PDS Analytics: Shop Performance Clusters")
st.caption("Fair Price Shops grouped into behavioural personas (K-Means) with DBSCAN anomaly flags.")

if f.empty:
    st.warning("No shops match the current filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Shops", f"{len(f):,}")
c2.metric("Avg monthly transactions", f"{f['avg_trans'].mean():,.0f}")
c3.metric("Avg portability share", f"{f['port_share'].mean():.1%}")
c4.metric("DBSCAN outliers", f"{int(f['is_outlier'].sum()):,}")

tab_map, tab_prof, tab_search = st.tabs(["Map", "Cluster profiles", "Shop search"])

# ---------------- Map ----------------
with tab_map:
    g = f[f["latitude"].between(15.8, 19.95) & f["longitude"].between(77.2, 81.8)]
    st.write(f"{len(g):,} of {len(f):,} shops have valid coordinates.")
    MAX_PTS = 3000
    if len(g) > MAX_PTS:
        keep = g[g["is_outlier"]]
        rest = g[~g["is_outlier"]].sample(MAX_PTS - len(keep), random_state=1)
        g = pd.concat([keep, rest])
        st.caption(f"Showing a sample of {MAX_PTS:,} shops (all outliers included). "
                   "Filter by district to see every shop.")
    fmap = folium.Map(location=[17.9, 79.6], zoom_start=7, tiles="OpenStreetMap")
    for r in g.itertuples():
        folium.CircleMarker(
            [r.latitude, r.longitude],
            radius=6 if r.is_outlier else 3,
            color="black" if r.is_outlier else PERSONA_COLORS[r.persona],
            weight=2 if r.is_outlier else 0,
            fill=True, fill_color=PERSONA_COLORS[r.persona], fill_opacity=0.75,
            tooltip=f"Shop {r.shopNo} | {r.distName} | {r.persona}"
                    + (" | OUTLIER" if r.is_outlier else ""),
        ).add_to(fmap)
    st_folium(fmap, height=550, use_container_width=True, returned_objects=[])
    st.markdown(" ".join(f"<span style='color:{c}'>●</span> {p}" for p, c in PERSONA_COLORS.items())
                + " &nbsp; ⚫ ringed = DBSCAN outlier", unsafe_allow_html=True)

# ---------------- Cluster profiles ----------------
with tab_prof:
    left, right = st.columns(2)
    counts = f["persona"].value_counts().reset_index()
    counts.columns = ["persona", "shops"]
    fig = px.bar(counts, x="persona", y="shops", color="persona",
                 color_discrete_map=PERSONA_COLORS, title="Shops per persona")
    fig.update_layout(showlegend=False, xaxis_title=None)
    left.plotly_chart(fig, use_container_width=True)

    samp = f.sample(min(6000, len(f)), random_state=1)
    fig2 = px.scatter(samp, x="pc1", y="pc2", color="persona",
                      color_discrete_map=PERSONA_COLORS, opacity=0.5,
                      title="Shops in PCA space (2D = 61% of variance)")
    right.plotly_chart(fig2, use_container_width=True)

    st.subheader("Mean of every feature by cluster")
    names = {0: "Urban Portability Hubs", 1: "Typical Rural Local Shops",
             2: "Volatile Low-Volume Shops", 3: "Semi-Urban Portability Shops"}
    p = prof.copy()
    p.insert(1, "persona", p["cluster"].map(names))
    st.dataframe(p.round(3), use_container_width=True, hide_index=True)

    dist_mix = (f.groupby(["distName", "persona"]).size().reset_index(name="shops"))
    fig3 = px.bar(dist_mix, x="distName", y="shops", color="persona",
                  color_discrete_map=PERSONA_COLORS, title="Persona mix by district")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- Shop search ----------------
with tab_search:
    st.write("Enter a shop number to compare it with its cluster average.")
    q = st.text_input("Shop number (shopNo)", placeholder="e.g. 1901001")
    if q.strip():
        if not q.strip().isdigit():
            st.error("Shop numbers are digits only.")
        else:
            hit = shops[shops["shopNo"] == int(q.strip())]
            if hit.empty:
                st.warning("Shop not found. Shops with fewer than 3 months of data "
                           "are not clustered.")
            else:
                s = hit.iloc[0]
                peers = shops[shops["cluster"] == s["cluster"]]
                a, b, c = st.columns(3)
                a.metric("Persona", s["persona"])
                b.metric("District / office", f"{s['distName']} / {s['officeName']}")
                c.metric("DBSCAN outlier", "YES" if s["is_outlier"] else "No")
                if s["is_outlier"]:
                    st.error("This shop's behaviour is unlike any dense cluster. Review it.")

                rows = []
                for col, label in COMPARE.items():
                    avg, sd = peers[col].mean(), peers[col].std()
                    z = (s[col] - avg) / sd if sd else 0.0
                    rows.append({"Metric": label, "This shop": round(s[col], 3),
                                 "Cluster average": round(avg, 3),
                                 "Difference %": round((s[col] / avg - 1) * 100, 1) if avg else None,
                                 "Z-score": round(z, 2),
                                 "Flag": "Unusual" if abs(z) > 3 else ""})
                comp = pd.DataFrame(rows)
                st.dataframe(comp, use_container_width=True, hide_index=True)

                bars = comp.assign(Shop=comp["Difference %"]).dropna(subset=["Shop"])
                figc = px.bar(bars, x="Metric", y="Shop", title="% difference from cluster average")
                st.plotly_chart(figc, use_container_width=True)

                trend = years[(years["shopNo"] == s["shopNo"]) & (years["distCode"] == s["distCode"])]
                if len(trend):
                    figt = px.line(trend, x="year", y="port_share", markers=True,
                                   title="This shop: portability share by year")
                    st.plotly_chart(figt, use_container_width=True)