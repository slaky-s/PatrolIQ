from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="PatrolIQ | Smart Safety Analytics", page_icon="🛡️", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "patroliq_models"
DATA_DIR = BASE_DIR / "data"
STREAMLIT_DATA_DIR = BASE_DIR / "patroliq_streamlit_data"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.hero {padding:1.3rem 1.5rem;border-radius:14px;background:linear-gradient(120deg,#102a43,#1f4e5f);color:white;margin-bottom:1rem;}
.hero h1 {margin:0;font-size:2.1rem}.hero p {margin:.4rem 0 0;color:#e5eef5}
</style>
""", unsafe_allow_html=True)

def first_existing(paths):
    return next((p for p in paths if p.exists()), None)

def load_csv(filename, folders=None):
    folders = folders or [MODEL_DIR, STREAMLIT_DATA_DIR, DATA_DIR, BASE_DIR]
    path = first_existing([folder / filename for folder in folders])
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None

def load_pickle(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:
        st.warning(f"Could not load {path.name}: {exc}")
        return None

def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def show_insight(title, bullets, note=None):
    """Display concise interpretation immediately after an analysis."""
    st.markdown(f"### 💡 {title}")
    for bullet in bullets:
        st.markdown(f"- {bullet}")
    if note:
        st.caption(note)

CHICAGO_COMMUNITY_AREAS = {
    1: "Rogers Park", 2: "West Ridge", 3: "Uptown", 4: "Lincoln Square",
    5: "North Center", 6: "Lake View", 7: "Lincoln Park", 8: "Near North Side",
    9: "Edison Park", 10: "Norwood Park", 11: "Jefferson Park", 12: "Forest Glen",
    13: "North Park", 14: "Albany Park", 15: "Portage Park", 16: "Irving Park",
    17: "Dunning", 18: "Montclare", 19: "Belmont Cragin", 20: "Hermosa",
    21: "Avondale", 22: "Logan Square", 23: "Humboldt Park", 24: "West Town",
    25: "Austin", 26: "West Garfield Park", 27: "East Garfield Park",
    28: "Near West Side", 29: "North Lawndale", 30: "South Lawndale",
    31: "Lower West Side", 32: "Loop", 33: "Near South Side", 34: "Armour Square",
    35: "Douglas", 36: "Oakland", 37: "Fuller Park", 38: "Grand Boulevard",
    39: "Kenwood", 40: "Washington Park", 41: "Hyde Park", 42: "Woodlawn",
    43: "South Shore", 44: "Chatham", 45: "Avalon Park", 46: "South Chicago",
    47: "Burnside", 48: "Calumet Heights", 49: "Roseland", 50: "Pullman",
    51: "South Deering", 52: "East Side", 53: "West Pullman", 54: "Riverdale",
    55: "Hegewisch", 56: "Garfield Ridge", 57: "Archer Heights", 58: "Brighton Park",
    59: "McKinley Park", 60: "Bridgeport", 61: "New City", 62: "West Elsdon",
    63: "Gage Park", 64: "Clearing", 65: "West Lawn", 66: "Chicago Lawn",
    67: "West Englewood", 68: "Englewood", 69: "Greater Grand Crossing",
    70: "Ashburn", 71: "Auburn Gresham", 72: "Beverly", 73: "Washington Heights",
    74: "Mount Greenwood", 75: "Morgan Park", 76: "O'Hare", 77: "Edgewater"
}

def build_enriched_cluster_data(geo_model, geo_clusters):
    """Join geography-only cluster labels to area and crime-share features."""
    if geo_model is None or geo_clusters is None:
        return None
    lat = next((c for c in ["Latitude", "latitude"] if c in geo_model.columns and c in geo_clusters.columns), None)
    lon = next((c for c in ["Longitude", "longitude"] if c in geo_model.columns and c in geo_clusters.columns), None)
    if not lat or not lon:
        return None
    cluster_col = next((c for c in geo_clusters.columns if c.lower() in {"kmeans_cluster", "geographic_kmeans_cluster"}), None)
    if not cluster_col:
        return None
    labels = geo_clusters[[lat, lon, cluster_col]].drop_duplicates([lat, lon]).copy()
    labels = labels.rename(columns={cluster_col: "Geographic_KMeans_Cluster"})
    data = geo_model.merge(labels, on=[lat, lon], how="inner")
    if data.empty:
        return None
    area_col = next((c for c in ["Community_Area", "community_area", "community_area_name", "area_name"] if c in data.columns), None)
    if area_col:
        numeric_area = pd.to_numeric(data[area_col], errors="coerce")
        if numeric_area.notna().mean() > 0.7:
            data["Community_Area_Name"] = numeric_area.round().map(CHICAGO_COMMUNITY_AREAS)
        else:
            data["Community_Area_Name"] = data[area_col].astype(str)
    return data

def cluster_area_crime_summary(enriched):
    """Summarize actual area labels, crime-share profile, and counts per cluster."""
    if enriched is None or "Geographic_KMeans_Cluster" not in enriched.columns:
        return None
    crime_cols = [c for c in enriched.columns if c.startswith("CrimeShare_")]
    rows = []
    for cluster_id, group in enriched.groupby("Geographic_KMeans_Cluster"):
        row = {
            "Geographic_Cluster": int(cluster_id),
            "Geographic_Cells": int(len(group)),
            "Total_Recorded_Crimes": int(group["Total_Crimes"].sum()) if "Total_Crimes" in group.columns else None,
            "Mean_Crimes_Per_Cell": round(float(group["Total_Crimes"].mean()), 2) if "Total_Crimes" in group.columns else None,
        }
        if "Community_Area_Name" in group.columns:
            areas = group["Community_Area_Name"].dropna()
            areas = areas[areas.astype(str).str.lower().ne("nan")]
            if not areas.empty:
                row["Most_Represented_Community_Area"] = areas.value_counts().index[0]
                row["Cells_In_Leading_Area"] = int(areas.value_counts().iloc[0])
        if crime_cols:
            means = group[crime_cols].mean(numeric_only=True).sort_values(ascending=False)
            if not means.empty:
                row["Dominant_Crime_Type_By_Mean_Share"] = means.index[0].replace("CrimeShare_", "").replace("_", " ")
                row["Mean_Crime_Share"] = round(float(means.iloc[0]), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Geographic_Cluster")

def cluster_bar(df, col):
    if df is None or col not in df.columns:
        return
    counts = df[col].astype(str).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title(f"Records by {col}")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Geographic cells")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.sidebar.title("🛡️ PatrolIQ")
st.sidebar.caption("Smart Safety Analytics Platform")
page = st.sidebar.radio("Navigate", [
    "Overview", "Geographic Crime Hotspot Clustering & Temporal Pattern Clustering", "Dimensionality Reduction"
])
st.sidebar.caption("Recorded-crime patterns are not validated individual safety or danger scores.")

geo_model = load_csv("geo_model.csv")
geo_clusters = load_csv("geographic_clustering_results.csv")
geo_profile = load_csv("geo_profile.csv")
kmeans_geo_profile = load_csv("kmeans_geographic_crime_profile.csv")
temporal_profile = load_csv("temporal_cluster_profile_k3.csv")
temporal_results = load_csv("temporal_clustering_results.csv")
pca_results = load_csv("pca_2d_results.csv")
pca_variance = load_csv("pca_explained_variance.csv")
pca_top_features = load_csv("pca_top_5_features.csv")
tsne_results = load_csv("tsne_results.csv")
cluster_comparison = load_csv("clustering_comparison.csv")
hierarchical_results = load_csv("hierarchical_geographic_results.csv")
dbscan_profile = load_csv("dbscan_cluster_profile.csv")
dbscan_geographic = load_csv("dbscan_geographic_results.csv")
dbscan_summary = load_csv("dbscan_geographic_summary.csv")
community_monthly = load_csv("community_area_crime_monthly.csv", [STREAMLIT_DATA_DIR, DATA_DIR, BASE_DIR])

if page == "Overview":
    hero("PatrolIQ", "Smart Safety Analytics Platform | Recorded-crime patterns across Chicago")
    st.write("Explore geographic clustering, temporal profiles, dimensionality reduction, and MLflow model tracking.")
    n_cells = len(geo_model) if geo_model is not None else (len(geo_clusters) if geo_clusters is not None else None)
    n_tsne = len(tsne_results) if tsne_results is not None else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Geographic cells", f"{n_cells:,}" if n_cells is not None else "—")
    c2.metric("Temporal clusters", "3" if temporal_profile is not None else "—")
    features = load_pickle("pca_feature_names.pkl")
    c3.metric("PCA features", str(len(features)) if features is not None else "—")
    c4.metric("t-SNE sample", f"{n_tsne:,}" if n_tsne is not None else "—")
    st.subheader("Project modules")
    a, b, c = st.columns(3)
    a.markdown("### 🗺️ Geographic analysis"); a.write("Explore cell-level counts and spatial clusters.")
    b.markdown("### 🕒 Temporal analysis"); b.write("Compare time-of-day and weekday/weekend profiles.")
    c.markdown("### 🧭 Dimensionality reduction"); c.write("Explore PCA variance and t-SNE embeddings.")
    show_insight(
        "Overview insights",
        [
            "The analysis works with 23,315 geographic cells and saved geographic, temporal, PCA, and t-SNE artifacts.",
            "These modules describe different aspects of recorded-crime patterns: location, timing, and similarity across features.",
            "The results are exploratory and should be used to guide further investigation, not to label people or places as dangerous."
        ],
        "Incident counts are not adjusted for population, foot traffic, exposure, or reporting rates."
    )
    st.warning("Counts are not adjusted for population, exposure, or reporting rates. Cluster labels are not danger ratings.")
    with st.expander("Artifact status"):
        names = ["geo_model.csv", "geographic_clustering_results.csv", "temporal_cluster_profile_k3.csv",
                 "pca_2d_results.csv", "pca_explained_variance.csv", "tsne_results.csv", "geographic_kmeans_k9.pkl"]
        st.dataframe(pd.DataFrame([{"Artifact": n, "Status": "Available" if (MODEL_DIR/n).exists() else "Not found"} for n in names]), use_container_width=True, hide_index=True)

elif page == "Crime Analysis":
    hero("Crime Analysis", "Explore available aggregated crime trends")
    if community_monthly is None:
        st.info("Add community_area_crime_monthly.csv to patroiq_streamlit_data/ to enable these charts.")
    else:
        st.dataframe(community_monthly.head(25), use_container_width=True, hide_index=True)
        count_col = next((c for c in ["incident_count", "Incident_Count", "Total_Crimes", "count"] if c in community_monthly.columns), None)
        date_col = next((c for c in ["month", "Month", "date", "Date"] if c in community_monthly.columns), None)
        area_col = next((c for c in ["community_area", "Community_Area", "area_name"] if c in community_monthly.columns), None)
        if count_col and date_col:
            d = community_monthly.copy()
            d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
            trend = d.dropna(subset=[date_col]).groupby(date_col, as_index=False)[count_col].sum()
            fig, ax = plt.subplots(figsize=(10, 4)); sns.lineplot(data=trend, x=date_col, y=count_col, ax=ax)
            ax.set_title("Recorded incidents over time"); ax.set_xlabel("Month"); ax.set_ylabel("Recorded incidents")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        if area_col and count_col:
            n = st.slider("Community areas to show", 5, 25, 10)
            top = community_monthly.groupby(area_col)[count_col].sum().nlargest(n).sort_values()
            fig, ax = plt.subplots(figsize=(9, 5)); top.plot(kind="barh", ax=ax)
            ax.set_title(f"Top {n} areas by recorded incident count"); ax.set_xlabel("Recorded incidents")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        show_insight(
            "Crime analysis insights",
            [
                "The monthly chart shows how recorded incident totals change across the available period; the top-area chart identifies areas with the largest totals in this aggregated file.",
                "The most represented category or period is not necessarily the one with the highest population-adjusted rate.",
                "Use these charts to identify patterns for further investigation, not to infer why crime changed or to assign safety labels."
            ]
        )

elif page == "Geographic Analysis":
    hero("Geographic Analysis", "Recorded-crime counts across geographic cells")
    source = geo_clusters if geo_clusters is not None else geo_model
    if source is None:
        st.info("Expected data: patroiq_models/geo_model.csv")
    else:
        count_col = next((c for c in ["Total_Crimes", "total_crimes", "incident_count"] if c in source.columns), None)
        lat_col = next((c for c in ["Latitude", "latitude", "lat"] if c in source.columns), None)
        lon_col = next((c for c in ["Longitude", "longitude", "lon", "lng"] if c in source.columns), None)
        filtered = source.copy()
        if count_col:
            lo, hi = float(source[count_col].min()), float(source[count_col].max())
            low, high = st.slider("Recorded-crime count range", lo, hi, (lo, hi))
            filtered = source[source[count_col].between(low, high)].copy()
        a, b, c = st.columns(3)
        a.metric("Cells shown", f"{len(filtered):,}")
        b.metric("Median count", f"{filtered[count_col].median():,.1f}" if count_col else "—")
        c.metric("Maximum count", f"{filtered[count_col].max():,.0f}" if count_col else "—")
        if lat_col and lon_col:
            m = filtered[[lat_col, lon_col]].rename(columns={lat_col:"lat", lon_col:"lon"}).dropna()
            st.map(m)
        st.dataframe(filtered.head(1000), use_container_width=True, hide_index=True)
        st.caption("Map points are geographic cells; counts are not population-adjusted risk estimates.")
        if count_col and not filtered.empty:
            top_cell = filtered.loc[filtered[count_col].idxmax()]
            show_insight(
                "Geographic analysis insights",
                [
                    f"The highest-count cell in the current filter contains {int(top_cell[count_col]):,} recorded incidents.",
                    "The map helps reveal spatial concentration and variation in recorded incident counts.",
                    "Counts are not adjusted for population, foot traffic, exposure, or reporting rates, so a high-count cell is not automatically a higher-risk area."
                ]
            )

elif page == "Geographic Crime Hotspot Clustering & Temporal Pattern Clustering":
    hero(
        "Explore Chicago's Crime Patterns",
        "Understand where recorded incidents group together, how locations differ, and when incidents are more common."
    )
    st.info(
        "How to read this page: zones describe groups of geographic cells; DBSCAN finds dense groups and isolated cells; "
        "temporal profiles describe when incidents tend to be recorded. These are descriptive patterns, not danger scores."
    )

    tab_zones, tab_density, tab_time, tab_method = st.tabs([
        "🗺️ Explore geographic zones", "📍 Density & isolated locations",
        "🕒 Time patterns", "ℹ️ How to interpret this"
    ])

    # -------------------- K-MEANS ZONES --------------------
    with tab_zones:
        st.subheader("Nine geographic zones")
        st.write(
            "Choose a zone to see where its cells are located and what the available summary says about recorded incidents. "
            "The zone numbers are labels only; a higher number does not mean more crime."
        )
        enriched_clusters = build_enriched_cluster_data(geo_model, geo_clusters)
        area_crime_summary = cluster_area_crime_summary(enriched_clusters)
        lat_col = next((c for c in ["Latitude", "latitude"] if geo_clusters is not None and c in geo_clusters.columns), None)
        lon_col = None
        if geo_clusters is not None:
            lon_col = next((c for c in ["Longitude", "longitude"] if c in geo_clusters.columns), None)
            km_col = next((c for c in ["KMeans_Cluster", "Geographic_KMeans_Cluster", "Geographic_Cluster"] if c in geo_clusters.columns), None)
        else:
            km_col = None

        if area_crime_summary is not None and not area_crime_summary.empty:
            summary = area_crime_summary.copy()
            cluster_id_col = "Geographic_Cluster"
            ids = sorted(summary[cluster_id_col].dropna().unique().tolist())
            selected_id = st.selectbox("Choose a geographic zone", ids, format_func=lambda x: f"Zone {int(x) + 1}")
            selected_row = summary.loc[summary[cluster_id_col] == selected_id].iloc[0]
            area_name = selected_row.get("Most_Represented_Community_Area", None)
            crime_name = selected_row.get("Dominant_Crime_Type_By_Mean_Share", None)
            total_crimes = selected_row.get("Total_Recorded_Crimes", None)
            cell_count = selected_row.get("Geographic_Cells", None)
            mean_count = selected_row.get("Mean_Crimes_Per_Cell", None)

            k1, k2, k3 = st.columns(3)
            k1.metric("Geographic cells in this zone", f"{int(cell_count):,}" if pd.notna(cell_count) else "—")
            k2.metric("Recorded incidents in these cells", f"{int(total_crimes):,}" if pd.notna(total_crimes) else "—")
            k3.metric("Average incidents per cell", f"{float(mean_count):,.1f}" if pd.notna(mean_count) else "—")

            sentences = []
            if pd.notna(area_name) and str(area_name).strip() and str(area_name).lower() != "nan":
                sentences.append(f"The most represented community area in this zone is **{area_name}**. The zone may also include cells from other areas.")
            if pd.notna(crime_name) and str(crime_name).strip() and str(crime_name).lower() != "nan":
                sentences.append(f"The strongest average crime-share category in the available model features is **{crime_name}**. This is a feature-based summary, not an incident-level count by crime type.")
            if pd.notna(total_crimes) and pd.notna(mean_count):
                sentences.append(f"The cells assigned to this zone contain **{int(total_crimes):,} recorded incidents** in the prepared dataset, averaging **{float(mean_count):.1f} per cell**.")
            if sentences:
                st.markdown("### What this zone tells us")
                for sentence in sentences:
                    st.markdown(f"- {sentence}")
            else:
                st.warning("The cluster summary exists, but area or crime details are missing. Check the matching geographic features.")

            if geo_clusters is not None and lat_col and lon_col and km_col:
                points = geo_clusters[[lat_col, lon_col, km_col]].dropna().copy()
                points[km_col] = pd.to_numeric(points[km_col], errors="coerce")
                points = points.dropna(subset=[km_col])
                points["Selected_Zone"] = np.where(points[km_col] == selected_id, "Selected zone", "Other zones")
                fig, ax = plt.subplots(figsize=(10, 6))
                other = points[points["Selected_Zone"] == "Other zones"]
                selected = points[points["Selected_Zone"] == "Selected zone"]
                if not other.empty:
                    ax.scatter(other[lon_col], other[lat_col], s=5, alpha=0.18, color="#a8b3bd", label="Other zones")
                if not selected.empty:
                    ax.scatter(selected[lon_col], selected[lat_col], s=12, alpha=0.9, color="#d1495b", label=f"Zone {int(selected_id)+1}")
                ax.set_title(f"Where Zone {int(selected_id)+1} sits in Chicago")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.legend(frameon=False)
                ax.grid(alpha=0.15)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption("Colored points are geographic cells. They are not official neighborhood boundaries or guaranteed contiguous polygons.")

            st.markdown("### Compare all nine zones")
            sort_choice = st.radio("Sort zones by", ["Most recorded incidents", "Highest average per cell", "Most geographic cells"], horizontal=True)
            sort_map = {"Most recorded incidents": "Total_Recorded_Crimes", "Highest average per cell": "Mean_Crimes_Per_Cell", "Most geographic cells": "Geographic_Cells"}
            sort_col = sort_map[sort_choice]
            ranked = summary.sort_values(sort_col, ascending=False).head(9)
            for _, row in ranked.iterrows():
                zone_id = int(row[cluster_id_col])
                area = row.get("Most_Represented_Community_Area", "Area not available")
                crime = row.get("Dominant_Crime_Type_By_Mean_Share", "Crime category not available")
                with st.container(border=True):
                    left, mid, right = st.columns([1.2, 2.4, 1.5])
                    left.markdown(f"**Zone {zone_id+1}**")
                    left.caption(f"{int(row['Geographic_Cells']):,} cells")
                    mid.markdown(f"**{area if pd.notna(area) else 'Area not available'}**")
                    mid.caption(f"Leading model share: {crime if pd.notna(crime) else 'not available'}")
                    val = row.get(sort_col, np.nan)
                    right.metric(sort_choice, f"{int(val):,}" if pd.notna(val) and sort_col != "Mean_Crimes_Per_Cell" else (f"{val:.1f}" if pd.notna(val) else "—"))
            with st.expander("See the underlying zone summary table"):
                st.dataframe(summary, use_container_width=True, hide_index=True)
        else:
            st.warning("Zone summaries could not be built. Check that geo_model.csv and geographic_clustering_results.csv share Latitude and Longitude values.")
            if geo_clusters is not None:
                st.caption("Available columns: " + ", ".join(map(str, geo_clusters.columns)))

    # -------------------- DBSCAN DENSITY --------------------
    with tab_density:
        st.subheader("Where are nearby cells grouped together?")
        st.write(
            "DBSCAN groups cells that have enough nearby neighbours. Grey points are cells that did not meet the selected density rule; "
            "they are not automatically unusual or dangerous."
        )
        if dbscan_geographic is not None:
            db_lat = next((c for c in ["Latitude", "latitude"] if c in dbscan_geographic.columns), None)
            db_lon = next((c for c in ["Longitude", "longitude"] if c in dbscan_geographic.columns), None)
            db_label = next((c for c in ["DBSCAN_Cluster", "dbscan_cluster"] if c in dbscan_geographic.columns), None)
            if db_lat and db_lon and db_label:
                dmap = dbscan_geographic[[db_lat, db_lon, db_label]].dropna().copy()
                dmap[db_label] = pd.to_numeric(dmap[db_label], errors="coerce")
                dmap = dmap.dropna(subset=[db_label])
                noise = dmap[db_label] == -1
                n_clusters = int(dmap.loc[~noise, db_label].nunique())
                n_noise = int(noise.sum())
                total_points = len(dmap)
                pct_noise = 100 * n_noise / total_points if total_points else 0
                m1, m2, m3 = st.columns(3)
                m1.metric("Dense groups found", f"{n_clusters:,}")
                m2.metric("Unclustered cells", f"{n_noise:,}")
                m3.metric("Cells left unclustered", f"{pct_noise:.1f}%")
                fig, ax = plt.subplots(figsize=(10, 6))
                clustered = dmap[~noise]
                if not clustered.empty:
                    ax.scatter(clustered[db_lon], clustered[db_lat], c=clustered[db_label], cmap="turbo", s=5, alpha=0.65, linewidths=0)
                if noise.any():
                    ax.scatter(dmap.loc[noise, db_lon], dmap.loc[noise, db_lat], c="#3d4852", s=9, alpha=0.85, label="Unclustered cells", linewidths=0)
                    ax.legend(frameon=False)
                ax.set_title("DBSCAN groups across Chicago")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.grid(alpha=0.15)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                if n_clusters > 0 and n_noise > 0:
                    st.markdown(
                        f"**In plain English:** DBSCAN grouped **{total_points-n_noise:,} of {total_points:,} cells** into **{n_clusters} groups**. "
                        f"Another **{n_noise:,} cells ({pct_noise:.1f}%)** did not have enough nearby neighbours under the current settings."
                    )
            else:
                st.warning("The DBSCAN results file is missing latitude, longitude, or DBSCAN_Cluster columns.")
            if dbscan_summary is not None and not dbscan_summary.empty:
                with st.expander("Explore the 10 largest DBSCAN groups"):
                    st.dataframe(dbscan_summary.head(10), use_container_width=True, hide_index=True)
        else:
            st.warning("DBSCAN result files were not found. Save dbscan_geographic_results.csv and dbscan_geographic_summary.csv in patroiq_models/.")
        st.caption("The current DBSCAN run found 73 groups and 2,365 unclustered cells. These numbers depend on the distance and neighbour settings.")

    # -------------------- TEMPORAL PATTERNS --------------------
    with tab_time:
        st.subheader("When are recorded incidents more common?")
        st.write("These profiles group geographic cells by their typical incident timing. Select a profile to read a short explanation.")
        if temporal_profile is None:
            st.warning("Temporal profile file not found: temporal_cluster_profile_k3.csv")
        else:
            profile = temporal_profile.copy()
            temporal_id_col = next((c for c in ["Temporal_Cluster", "Cluster", "cluster"] if c in profile.columns), None)
            time_cols = [c for c in ["TimeShare_Morning", "TimeShare_Afternoon", "TimeShare_Evening", "TimeShare_Night"] if c in profile.columns]
            day_cols = [c for c in profile.columns if c.startswith("DayShare_")]
            if temporal_id_col and time_cols:
                profile["Profile_Label"] = profile[temporal_id_col].map({0: "Night & weekend-oriented", 1: "Morning & daytime-oriented", 2: "Evening-oriented"}).fillna(profile[temporal_id_col].astype(str).radd("Profile "))
                labels = profile["Profile_Label"].tolist()
                selected_profile = st.selectbox("Choose a time pattern", labels)
                row = profile.loc[profile["Profile_Label"] == selected_profile].iloc[0]
                share_vals = pd.to_numeric(row[time_cols], errors="coerce")
                best_col = share_vals.idxmax() if share_vals.notna().any() else None
                best_period = best_col.replace("TimeShare_", "").lower() if best_col else "not available"
                count_col = next((c for c in ["Total_Incidents", "Total_Crimes", "Incident_Count", "Total_Recorded_Crimes"] if c in profile.columns), None)
                a, b, c = st.columns(3)
                a.metric("Time pattern", selected_profile)
                b.metric("Largest time-of-day share", best_period.title())
                c.metric("Cells in this profile", f"{int(row['Geographic_Cells']):,}" if "Geographic_Cells" in profile.columns and pd.notna(row.get("Geographic_Cells")) else "—")
                if best_col:
                    st.markdown(f"### What this means")
                    st.write(f"Across this profile, **{best_period}** has the largest average share of incidents among the time-of-day categories included in the model.")
                d = profile[[temporal_id_col] + time_cols].melt(id_vars=temporal_id_col, var_name="Time of day", value_name="Average share")
                d["Time of day"] = d["Time of day"].str.replace("TimeShare_", "", regex=False)
                d["Profile"] = d[temporal_id_col].map({0: "Night & weekend", 1: "Morning & daytime", 2: "Evening"}).fillna(d[temporal_id_col].astype(str))
                fig, ax = plt.subplots(figsize=(9, 4.5))
                sns.barplot(data=d, x="Time of day", y="Average share", hue="Profile", ax=ax)
                ax.set_title("Typical timing of each profile")
                ax.set_ylabel("Average share (0–1)")
                ax.set_xlabel("")
                ax.legend(title="Pattern", frameon=False)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                if day_cols:
                    with st.expander("Explore weekday and weekend differences"):
                        days = profile[[temporal_id_col] + day_cols].melt(id_vars=temporal_id_col, var_name="Day", value_name="Average share")
                        days["Day"] = days["Day"].str.replace("DayShare_", "", regex=False)
                        days["Profile"] = days[temporal_id_col].map({0: "Night & weekend", 1: "Morning & daytime", 2: "Evening"}).fillna(days[temporal_id_col].astype(str))
                        fig, ax = plt.subplots(figsize=(10, 4.5))
                        sns.barplot(data=days, x="Day", y="Average share", hue="Profile", ax=ax)
                        ax.set_title("Average weekday share by profile")
                        ax.tick_params(axis="x", rotation=25)
                        ax.set_xlabel("")
                        ax.legend(title="Pattern", frameon=False)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                with st.expander("See profile data and cell assignments"):
                    st.dataframe(profile, use_container_width=True, hide_index=True)
                    if temporal_results is not None:
                        st.dataframe(temporal_results.head(500), use_container_width=True, hide_index=True)
            else:
                st.warning("Temporal profile columns did not match the expected format.")

        # Seasonal/monthly trend if the existing aggregate file supports it.
        if community_monthly is not None:
            count_col = next((c for c in ["incident_count", "Incident_Count", "Total_Crimes", "count"] if c in community_monthly.columns), None)
            date_col = next((c for c in ["month", "Month", "date", "Date"] if c in community_monthly.columns), None)
            if count_col and date_col:
                trend = community_monthly.copy()
                trend[date_col] = pd.to_datetime(trend[date_col], errors="coerce")
                trend[count_col] = pd.to_numeric(trend[count_col], errors="coerce")
                trend = trend.dropna(subset=[date_col, count_col]).groupby(date_col, as_index=False)[count_col].sum().sort_values(date_col)
                if not trend.empty:
                    with st.expander("Explore month-by-month recorded incident trends"):
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.lineplot(data=trend, x=date_col, y=count_col, marker="o", ax=ax)
                        ax.set_title("Recorded incidents over time")
                        ax.set_xlabel("")
                        ax.set_ylabel("Recorded incidents")
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                        st.caption("Seasonal changes describe the data period and reporting patterns; they do not explain the cause of changes.")

    # -------------------- EXPLAINER --------------------
    with tab_method:
        st.subheader("What each method tells you")
        with st.container(border=True):
            st.markdown("**K-Means — nine geographic zones**")
            st.write("Splits the geographic cells into a fixed number of groups so you can compare parts of the city. It does not by itself draw official or contiguous boundaries.")
        with st.container(border=True):
            st.markdown("**DBSCAN — density groups and isolated cells**")
            st.write("Groups cells with enough nearby neighbours. Results depend on the distance threshold and minimum-neighbour setting.")
        with st.container(border=True):
            st.markdown("**Hierarchical clustering — relationships between cells**")
            st.write("A dendrogram shows which sampled cells or groups merge earlier or later. It is a relationship diagram, not a crime ranking.")
        with st.container(border=True):
            st.markdown("**Temporal clustering — typical timing patterns**")
            st.write("Groups geographic cells with similar timing profiles. The current model has three exploratory patterns: night/weekend-oriented, morning/daytime-oriented, and evening-oriented.")
        st.warning("Recorded incident counts are not adjusted for population, foot traffic, exposure, or reporting rates. Do not interpret clusters as validated danger ratings.")

elif page == "Temporal Analysis":
    hero("Temporal Analysis", "Temporal profiles across geographic cells")
    if temporal_profile is None:
        st.info("Expected data: temporal_cluster_profile_k3.csv")
    else:
        st.write("These profiles summarize incident timing across geographic cells; they are not individual incident clusters or validated dangerous-time estimates.")
        st.dataframe(temporal_profile, use_container_width=True, hide_index=True)
        show_insight(
            "Temporal clustering insights",
            [
                "Cluster 0 is night-oriented: its mean night share is 37.1%, and its weekend share is 43.9%.",
                "Cluster 1 is more morning-oriented: its mean morning share is 28.7%, with a weekend share of 17.9%.",
                "Cluster 2 is evening-oriented: its mean evening share is 37.0%, and it contains the largest total incident count of the three groups.",
                "The K=3 silhouette score is 0.0887, so the profiles are weakly separated and should be interpreted as exploratory."
            ],
            "These are average timing profiles of geographic cells, not predictions that a time or location is unsafe."
        )
        time_cols = [c for c in ["TimeShare_Morning", "TimeShare_Afternoon", "TimeShare_Evening", "TimeShare_Night"] if c in temporal_profile.columns]
        if time_cols and "Temporal_Cluster" in temporal_profile.columns:
            d = temporal_profile[["Temporal_Cluster"]+time_cols].melt(id_vars="Temporal_Cluster", var_name="Time_Period", value_name="Mean_Share")
            d["Time_Period"] = d["Time_Period"].str.replace("TimeShare_", "", regex=False)
            fig, ax = plt.subplots(figsize=(10, 5)); sns.barplot(data=d, x="Time_Period", y="Mean_Share", hue="Temporal_Cluster", ax=ax)
            ax.set_title("Time-of-day profile by temporal cluster"); ax.set_ylabel("Mean share of recorded incidents")
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        day_cols = [c for c in temporal_profile.columns if c.startswith("DayShare_")]
        if day_cols and "Temporal_Cluster" in temporal_profile.columns:
            d = temporal_profile[["Temporal_Cluster"]+day_cols].melt(id_vars="Temporal_Cluster", var_name="Day", value_name="Mean_Share")
            d["Day"] = d["Day"].str.replace("DayShare_", "", regex=False)
            fig, ax = plt.subplots(figsize=(11, 5)); sns.barplot(data=d, x="Day", y="Mean_Share", hue="Temporal_Cluster", ax=ax)
            ax.set_title("Day-of-week profile by temporal cluster"); ax.tick_params(axis="x", rotation=30)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        if temporal_results is not None:
            cluster_bar(temporal_results, "Temporal_Cluster")
            st.dataframe(temporal_results.head(1000), use_container_width=True, hide_index=True)

elif page == "Dimensionality Reduction":
    hero("Dimensionality Reduction", "Find simpler ways to look at many crime-pattern details")
    st.markdown("""
    ### What does this mean in everyday language?

    Imagine each Chicago map cell has a long report card: how many crimes were recorded, which crime types appeared,
    what times incidents happened, and what days they happened. Comparing every detail at once is difficult.

    **Dimensionality reduction** tries to make those report cards easier to explore:
    - **PCA** combines related details into a smaller set of summary scores.
    - **t-SNE** puts similar report cards near each other on a 2D picture.

    **Important:** the dots are geographic cells summarized by features, not individual people or individual crime incidents.
    """)
    tab_pca, tab_tsne, tab_guide = st.tabs(["PCA: simplify the details", "t-SNE: find similar patterns", "Plain-English guide"])

    with tab_pca:
        st.subheader("PCA: how much detail remains after simplifying?")
        st.info("Think of PCA like summarizing a long report into a few headline points. The percentage tells us how much of the original variation those summaries still capture.")
        if pca_variance is None:
            st.info("Expected data: pca_explained_variance.csv in the PatrolIQ model/data folder.")
        else:
            var_col = next((c for c in ["Explained_Variance_Ratio", "explained_variance_ratio", "ExplainedVarianceRatio"] if c in pca_variance.columns), None)
            cum_col = next((c for c in ["Cumulative_Variance_Ratio", "Cumulative_Explained_Variance_Ratio", "cumulative_variance_ratio"] if c in pca_variance.columns), None)
            if var_col:
                ratios = pd.to_numeric(pca_variance[var_col], errors="coerce").dropna().to_numpy()
                cumulative = np.cumsum(ratios)
                if cum_col and pca_variance[cum_col].notna().any():
                    saved_cumulative = pd.to_numeric(pca_variance[cum_col], errors="coerce").to_numpy()
                    if len(saved_cumulative) == len(ratios) and np.isfinite(saved_cumulative).any():
                        cumulative = np.nan_to_num(saved_cumulative, nan=0.0)
                pc1_2 = float(np.sum(ratios[:2]) * 100) if len(ratios) >= 2 else float(np.sum(ratios) * 100)
                pc1_3 = float(np.sum(ratios[:3]) * 100)
                n70 = int(np.argmax(cumulative >= .70) + 1) if np.any(cumulative >= .70) else None
                n80 = int(np.argmax(cumulative >= .80) + 1) if np.any(cumulative >= .80) else None
                a, b, c, d = st.columns(4)
                a.metric("First 2 summary scores", f"{pc1_2:.2f}%")
                b.metric("First 3 summary scores", f"{pc1_3:.2f}%")
                c.metric("Scores needed to keep 70%", str(n70) if n70 else "Not reached")
                d.metric("Scores needed to keep 80%", str(n80) if n80 else "Not reached")

                fig, ax = plt.subplots(figsize=(10, 4.5))
                component_numbers = np.arange(1, len(cumulative) + 1)
                ax.plot(component_numbers, cumulative * 100, marker=".", linewidth=2)
                ax.axhline(70, linestyle="--", label="70% target")
                ax.axhline(80, linestyle=":", label="80% target")
                ax.set_ylim(0, 100)
                ax.set_xlabel("Number of summary scores kept")
                ax.set_ylabel("Share of original data variation retained (%)")
                ax.set_title("How much original detail is kept as we add summary scores")
                ax.grid(alpha=.25)
                ax.legend()
                fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
                if pc1_3 < 70:
                    st.warning(
                        f"The target of explaining 70–80% of variance with only 2–3 components is not met. "
                        f"The first three explain {pc1_3:.2f}%; {n70 or 'More than the available'} components are needed for 70%."
                    )
                else:
                    st.success("The first three components meet the 70% variance target in the saved results.")
            else:
                st.warning("The variance file was found, but its explained-variance column was not recognized.")

        st.subheader("PCA picture: each dot is one map cell")
        st.caption("PC1 and PC2 are not street coordinates. They are two calculated summary scores made by combining many columns from the data.")
        if pca_results is None or not {"PC1", "PC2"}.issubset(pca_results.columns):
            st.info("Expected data: pca_2d_results.csv with PC1 and PC2 columns.")
        else:
            plot_df = pca_results.copy()
            color_options = [c for c in ["KMeans_Cluster", "Geographic_Cluster", "Temporal_Cluster", "Community_Area"] if c in plot_df.columns]
            color_by = st.selectbox("What should the colors represent?", ["No grouping"] + color_options, key="pca_color_by")
            if color_by != "No grouping":
                st.caption("The colors show an existing label in the data. A K-Means cluster is just a computer-assigned group number—not a crime grade or danger level.")
            max_points = min(10000, len(plot_df))
            default_points = min(5000, max_points)
            n_points = st.slider("PCA points to display", min_value=min(500, max_points), max_value=max_points, value=default_points, step=500 if max_points >= 1000 else 1, key="pca_point_count") if max_points > 500 else max_points
            if n_points < len(plot_df):
                plot_df = plot_df.sample(n=n_points, random_state=42)
            fig, ax = plt.subplots(figsize=(10, 6))
            if color_by != "No grouping":
                sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=color_by, palette="tab10", s=22, alpha=.7, linewidth=0, ax=ax)
                ax.legend(title=color_by.replace("_", " "), bbox_to_anchor=(1.02, 1), loc="upper left")
            else:
                ax.scatter(plot_df["PC1"], plot_df["PC2"], s=18, alpha=.65)
            ax.set_title("PCA: geographic cells projected onto the first two components")
            ax.set_xlabel("Summary score 1 (PC1)")
            ax.set_ylabel("Summary score 2 (PC2)")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.caption("PC axes are combinations of the original features, not latitude and longitude. The first two PCs retain only a small share of total variance.")

        st.subheader("Where are these patterns in Chicago?")
        st.markdown("This second view puts geographic cells back onto their real latitude/longitude positions. Unlike the PCA scatter plot above, this is a location map.")
        st.caption("If the saved PCA results contain coordinates or a reliable cell ID, the map can show PCA-derived values. Otherwise, the map uses the existing geographic K-Means labels and clearly identifies them as geographic clusters.")

        map_df = None
        map_color = None
        map_color_label = None
        if pca_results is not None:
            # Use PCA-derived map colors only when the PCA rows can be matched to real coordinates.
            pca_lat = next((c for c in ["Latitude", "latitude", "lat"] if c in pca_results.columns), None)
            pca_lon = next((c for c in ["Longitude", "longitude", "lon", "lng"] if c in pca_results.columns), None)
            if pca_lat and pca_lon:
                candidate = pca_results.copy()
                candidate["_map_lat"] = pd.to_numeric(candidate[pca_lat], errors="coerce")
                candidate["_map_lon"] = pd.to_numeric(candidate[pca_lon], errors="coerce")
                candidate = candidate.dropna(subset=["_map_lat", "_map_lon"])
                score_col = next((c for c in ["PC1", "PC2"] if c in candidate.columns), None)
                if not candidate.empty and score_col:
                    map_df = candidate.rename(columns={"_map_lat": "Latitude", "_map_lon": "Longitude"})
                    map_color = score_col
                    map_color_label = f"PCA summary score ({score_col})"

        if map_df is None and geo_clusters is not None:
            map_lat = next((c for c in ["Latitude", "latitude", "lat"] if c in geo_clusters.columns), None)
            map_lon = next((c for c in ["Longitude", "longitude", "lon", "lng"] if c in geo_clusters.columns), None)
            map_cluster = next((c for c in ["KMeans_Cluster", "Geographic_KMeans_Cluster", "Geographic_Cluster"] if c in geo_clusters.columns), None)
            if map_lat and map_lon:
                map_df = geo_clusters.copy()
                map_df["Latitude"] = pd.to_numeric(map_df[map_lat], errors="coerce")
                map_df["Longitude"] = pd.to_numeric(map_df[map_lon], errors="coerce")
                map_df = map_df.dropna(subset=["Latitude", "Longitude"])
                if map_cluster:
                    map_df["Map_Group"] = map_df[map_cluster].astype(str)
                    map_color = "Map_Group"
                    map_color_label = "Existing geographic cluster"
                else:
                    map_color = None
                    map_color_label = "Recorded-crime locations"

        if map_df is None or map_df.empty or not {"Latitude", "Longitude"}.issubset(map_df.columns):
            st.info("To show the Chicago map, include geographic_clustering_results.csv with Latitude and Longitude columns. The PCA plot above will still work without it.")
        else:
            try:
                import plotly.express as px
                map_max = min(10000, len(map_df))
                map_default = min(5000, map_max)
                map_n = st.slider("Map dots to display", min_value=min(500, map_max), max_value=map_max, value=map_default, step=500 if map_max >= 1000 else 1, key="pca_geo_map_points") if map_max > 500 else map_max
                if map_n < len(map_df):
                    map_df = map_df.sample(n=map_n, random_state=42)
                hover_cols = [c for c in ["Total_Crimes", "KMeans_Cluster", "DBSCAN_Cluster", "PC1", "PC2"] if c in map_df.columns]
                fig_map = px.scatter_mapbox(
                    map_df, lat="Latitude", lon="Longitude", color=map_color,
                    hover_data=hover_cols, zoom=9, height=560,
                    title="Chicago map: actual geographic positions",
                    color_continuous_scale="Viridis" if map_color and pd.api.types.is_numeric_dtype(map_df[map_color]) else None,
                )
                fig_map.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 45, "l": 0, "b": 0}, legend_title_text=map_color_label or "Locations")
                st.plotly_chart(fig_map, use_container_width=True)
                st.caption(f"Map color: {map_color_label or 'all recorded geographic cells'}. Dots are sampled for readability; they represent geographic cells, not individual crime incidents.")
                if map_color_label and "PCA" not in map_color_label:
                    st.info("This map is colored by existing geographic cluster labels, not PCA scores, because the saved PCA rows did not provide coordinates that could be matched reliably.")
            except Exception as exc:
                st.warning(f"The geographic map could not be rendered. Check that Plotly is installed and the coordinate columns are valid. Details: {exc}")

        st.subheader("Which original details contribute to the PCA summaries?")
        st.caption("This table shows which data columns contribute strongly to the calculated summary scores. It does not mean those columns cause crime.")
        if pca_top_features is None:
            st.info("Expected data: pca_top_5_features.csv.")
        else:
            st.dataframe(pca_top_features, use_container_width=True, hide_index=True)
            st.caption("In plain language: these are ingredients used to build the summaries—not a ranking of what causes crime or what a prediction model relies on most.")
        show_insight(
            "PCA takeaways",
            [
                "PCA is like turning a long report card into a few summary scores. It tries to keep the biggest differences between map cells.",
                "In this run, the first two scores preserve only 11.07% of the original variation; the first three preserve 15.40%.",
                "Two or three scores are not enough to represent this dataset well. About 27 scores are needed to preserve around 70% of its variation.",
                "Weekend share, average severity, battery share, night share, and theft share were among the strongest combined loadings for the first two components."
            ],
            "A feature's PCA loading describes its contribution to a component; it does not establish that the feature causes crime."
        )

    with tab_tsne:
        st.subheader("t-SNE picture: which map cells have similar patterns?")
        st.info("Think of this as a similarity map made from the data. Dots close together have more similar recorded-crime feature profiles in this picture; dots far apart look less similar. Their positions are not real locations on a Chicago map.")
        if tsne_results is None:
            st.info("Expected data: tsne_results.csv.")
        elif not {"TSNE_1", "TSNE_2"}.issubset(tsne_results.columns):
            st.warning("The t-SNE file was found, but TSNE_1 and TSNE_2 columns were not found.")
        else:
            st.caption("Each dot represents a sampled geographic cell. Nearby dots have more similar feature patterns in this embedding; the axes are not geographic coordinates.")
            plot_df = tsne_results.copy()
            color_options = [c for c in ["KMeans_Cluster", "Geographic_Cluster", "Temporal_Cluster", "Community_Area"] if c in plot_df.columns]
            color_by = st.selectbox("Color dots using", ["No grouping"] + color_options, key="tsne_color_by")
            if color_by != "No grouping":
                st.caption("Colors show a label already in the dataset. Cluster 1 simply means the algorithm assigned that group number; it does not mean first place, highest crime, or highest danger.")
            max_points = min(5000, len(plot_df))
            default_points = max_points
            n_points = st.slider("t-SNE points to display", min_value=min(500, max_points), max_value=max_points, value=default_points, step=500 if max_points >= 1000 else 1, key="tsne_point_count") if max_points > 500 else max_points
            if n_points < len(plot_df):
                plot_df = plot_df.sample(n=n_points, random_state=42)
            fig, ax = plt.subplots(figsize=(10, 6))
            if color_by != "No grouping":
                sns.scatterplot(data=plot_df, x="TSNE_1", y="TSNE_2", hue=color_by, palette="tab10", s=20, alpha=.72, linewidth=0, ax=ax)
                ax.legend(title=color_by.replace("_", " "), bbox_to_anchor=(1.02, 1), loc="upper left")
            else:
                ax.scatter(plot_df["TSNE_1"], plot_df["TSNE_2"], s=18, alpha=.7)
            ax.set_title("t-SNE: similarity view of crime-pattern features")
            ax.set_xlabel("Similarity-map direction 1")
            ax.set_ylabel("Similarity-map direction 2")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.caption("Visible separation is exploratory. It does not prove that the groups are statistically distinct or geographically far apart.")
            st.metric("Points shown", f"{len(plot_df):,}")
            if color_by != "No grouping":
                st.write("**How many dots are in each displayed group?**")
                st.caption("These counts describe the dots currently shown, not a ranking of crime severity or danger.")
                counts = plot_df[color_by].astype(str).value_counts().rename_axis(color_by).reset_index(name="Points")
                st.dataframe(counts, use_container_width=True, hide_index=True)
        show_insight(
            "t-SNE takeaways",
            [
                "The saved t-SNE picture uses a sample of 5,000 map cells. The computer first summarized the data with 27 PCA scores, then arranged similar profiles in 2D.",
                "Nearby dots mean similar data patterns in this picture—not nearby streets or neighborhoods.",
                "If different colors overlap, the patterns are not cleanly separated in this view. t-SNE does not prove distinct crime types or groups exist."
            ],
            "t-SNE is mainly a visualization tool; the apparent size and distance between groups can change with its settings."
        )

    with tab_guide:
        st.subheader("The short version")
        st.markdown(
            "- **PCA** summarizes many numeric features into components and reports how much variation those components retain.\n"
            "- **t-SNE** places similar feature profiles near each other in a 2D visualization. It does not create verified crime categories by itself.\n"
            "- **Feature contribution is not causation.** PCA loadings are not the same as feature importance from a predictive model.\n"
            "- **The original target is not met:** this saved PCA run does not retain 70–80% of variance in only 2–3 components.\n"
            "- The existing t-SNE file contains geographic-cell profiles, so this view compares patterns across cells rather than separating individual incident types directly."
        )
        st.warning("Recorded-crime counts are not adjusted for population, foot traffic, reporting rates, or exposure. Do not interpret clusters as danger scores.")

elif page == "Model Evaluation":
    hero("Model Evaluation", "Available clustering evaluation artifacts")
    if cluster_comparison is not None:
        st.subheader("Clustering comparison")
        st.dataframe(cluster_comparison, use_container_width=True, hide_index=True)
    else: st.info("clustering_comparison.csv not found.")
    if hierarchical_results is not None:
        st.subheader("Hierarchical clustering sample")
        st.caption("Hierarchical labels are from the reproducible 2,000-cell sample, not all geographic cells.")
        st.dataframe(hierarchical_results.head(1000), use_container_width=True, hide_index=True)
    if geo_profile is not None:
        st.subheader("Geographic profile")
        st.dataframe(geo_profile.head(1000), use_container_width=True, hide_index=True)
    st.info("Temporal K-Means K=3 silhouette: 0.0887; Davies–Bouldin: 2.5012. Treat the temporal groups as exploratory profiles.")
    st.subheader("Requirement check")
    st.warning(
        "The target silhouette score above 0.5 has not been achieved by the newer geographic K-Means candidate "
        "(0.3946). The dashboard should report this transparently."
    )
    show_insight(
        "Model evaluation insights",
        [
            "Use silhouette and Davies–Bouldin together with cluster sizes and interpretability; no single metric determines the best model.",
            "Geographic K-Means K=9 is the practical candidate from the newer latitude/longitude-only experiment (silhouette 0.3946; Davies–Bouldin 0.7696).",
            "Hierarchical Ward clustering was evaluated on a 2,000-cell sample; its labels should not be treated as predictions for all 23,315 cells.",
            "The older feature-based K-Means result is a different experiment and should not be directly compared with the newer geography-only model without noting the feature differences."
        ]
    )

elif page == "MLflow Registry":
    hero("MLflow Model Registry", "Registered models and sample geographic-cluster prediction")
    try:
        import mlflow
        from mlflow import MlflowClient
        client = MlflowClient()
        registered = client.search_registered_models()
        rows = []
        for model in registered:
            for version in client.search_model_versions(f"name='{model.name}'"):
                rows.append({"Model":model.name,"Version":version.version,"Status":version.status,"Run ID":version.run_id})
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else: st.info("No registered models found for the current MLflow tracking URI.")
        st.caption(f"Tracking URI: {mlflow.get_tracking_uri()}")
        st.subheader("Test a geographic cluster prediction")
        lat = st.number_input("Latitude", value=41.88, format="%.6f")
        lon = st.number_input("Longitude", value=-87.63, format="%.6f")
        if st.button("Predict cluster"):
            scaler = load_pickle("geographic_kmeans_scaler.pkl")
            if scaler is None:
                st.error("Missing geographic_kmeans_scaler.pkl in patroiq_models/.")
            else:
                try:
                    model = mlflow.pyfunc.load_model("models:/PatrolIQ_Geographic_KMeans/1")
                    inputs = pd.DataFrame({"Latitude":[lat],"Longitude":[lon]})
                    scaled = scaler.transform(inputs[["Latitude","Longitude"]])
                    prediction = model.predict(scaled)
                    st.success(f"Predicted geographic cluster: {int(prediction[0])}")
                    st.caption("Cluster labels are not danger ratings.")
                    show_insight(
                        "MLflow registry insights",
                        [
                            "PatrolIQ_Geographic_KMeans version 1 was registered with READY status and passed the sample prediction test.",
                            "The model assigns input coordinates to one of nine geographic clusters; it does not predict crime probability.",
                            "Keep the matching geographic scaler with the registered model and use the same feature order during inference."
                        ]
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.info("Ensure this app uses the same MLflow tracking URI/database as the notebook.")
    except Exception as exc:
        st.error(f"Could not read MLflow registry: {exc}")
        st.info("Other dashboard pages can still be used if MLflow is unavailable.")

st.divider()
st.caption("PatrolIQ | Exploratory recorded-crime analytics. Cluster labels and incident counts are not validated individual safety predictions.")
