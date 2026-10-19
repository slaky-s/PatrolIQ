# app.py
# ============================================================
# PatrolIQ — Smart Safety Analytics Platform
# Streamlit Dashboard
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PatrolIQ",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "patroliq_models")
VISUALIZATION_DIR = os.path.join(BASE_DIR, "mlflow_visualizations")


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_models():

    scaler = joblib.load(
        os.path.join(
            MODEL_DIR,
            "standard_scaler.pkl"
        )
    )

    kmeans = joblib.load(
        os.path.join(
            MODEL_DIR,
            "kmeans_k8.pkl"
        )
    )

    dbscan = joblib.load(
        os.path.join(
            MODEL_DIR,
            "dbscan_eps7.pkl"
        )
    )

    pca = joblib.load(
        os.path.join(
            MODEL_DIR,
            "pca_model.pkl"
        )
    )

    pca_metadata = joblib.load(
        os.path.join(
            MODEL_DIR,
            "pca_metadata.pkl"
        )
    )

    feature_names = joblib.load(
        os.path.join(
            MODEL_DIR,
            "feature_names.pkl"
        )
    )

    return (
        scaler,
        kmeans,
        dbscan,
        pca,
        pca_metadata,
        feature_names
    )


@st.cache_data
def load_data():

    geo_model = pd.read_csv(
        os.path.join(
            MODEL_DIR,
            "geo_model.csv"
        )
    )

    geo_profile = pd.read_csv(
        os.path.join(
            MODEL_DIR,
            "geo_profile.csv"
        )
    )

    kmeans_profile = pd.read_csv(
        os.path.join(
            MODEL_DIR,
            "kmeans_cluster_profile.csv"
        )
    )

    dbscan_profile = pd.read_csv(
        os.path.join(
            MODEL_DIR,
            "dbscan_cluster_profile.csv"
        )
    )

    comparison = pd.read_csv(
        os.path.join(
            MODEL_DIR,
            "clustering_comparison.csv"
        )
    )

    return (
        geo_model,
        geo_profile,
        kmeans_profile,
        dbscan_profile,
        comparison
    )


(
    scaler,
    kmeans,
    dbscan,
    pca,
    pca_metadata,
    feature_names
) = load_models()


(
    geo_model,
    geo_profile,
    kmeans_profile,
    dbscan_profile,
    comparison
) = load_data()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ PatrolIQ")
st.subheader(
    "Smart Safety Analytics Platform"
)

st.markdown(
    """
    **PatrolIQ** explores reported crime patterns using geographic,
    temporal, clustering and dimensionality-reduction analytics.

    The platform is designed as a **decision-support analytics tool**
    for exploring aggregate reported-crime patterns.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("PatrolIQ")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Crime Analysis",
        "Geographic Analysis",
        "Clustering",
        "Dimensionality Reduction",
        "Model Evaluation"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Data basis: 500,000 recent records "
    "from the Chicago crime dataset."
)

st.sidebar.caption(
    "Geographic ML cells: 23,315"
)


# ============================================================
# COMMON CALCULATIONS
# ============================================================

total_geographic_records = int(
    geo_profile["Total_Crimes"].sum()
)

geographic_cells = len(
    geo_profile
)

modeling_cells = len(
    geo_model
)

avg_crimes_per_cell = (
    geo_profile["Total_Crimes"].mean()
)

avg_severity = (
    geo_profile["Avg_Severity"].mean()
)

avg_arrest_rate = (
    geo_profile["Arrest_Rate"].mean()
)

avg_domestic_rate = (
    geo_profile["Domestic_Rate"].mean()
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Overview")

    st.info(
        f"""
        The dashboard is based on the recent 500,000-record analytical
        sample. Geographic modeling uses {modeling_cells:,} cells with
        at least 5 reported incidents.
        """
    )

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Geographic Records",
            f"{total_geographic_records:,}"
        )

    with col2:
        st.metric(
            "Geographic Cells",
            f"{geographic_cells:,}"
        )

    with col3:
        st.metric(
            "ML Modeling Cells",
            f"{modeling_cells:,}"
        )

    with col4:
        st.metric(
            "Avg Crimes / Cell",
            f"{avg_crimes_per_cell:.2f}"
        )

    st.divider()

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Severity",
            f"{avg_severity:.2f}"
        )

    with col2:
        st.metric(
            "Average Arrest Rate",
            f"{avg_arrest_rate * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Average Domestic Rate",
            f"{avg_domestic_rate * 100:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # TOP CRIME TYPES
    # --------------------------------------------------------

    st.subheader(
        "Dominant Crime Patterns Across Geographic Cells"
    )

    crime_counts = (
        geo_profile["Dominant_Crime"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    crime_counts.columns = [
        "Crime Type",
        "Geographic Cells"
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=crime_counts,
        x="Geographic Cells",
        y="Crime Type",
        ax=ax
    )

    ax.set_title(
        "Top Dominant Crime Types by Geographic Cell"
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


# ============================================================
# PAGE 2 — CRIME ANALYSIS
# ============================================================

elif page == "Crime Analysis":

    st.header("Crime Analysis")

    st.markdown(
        """
        Explore the distribution of dominant reported-crime patterns
        across the geographic cells used by PatrolIQ.
        """
    )

    # --------------------------------------------------------
    # Crime type
    # --------------------------------------------------------

    crime_counts = (
        geo_profile["Dominant_Crime"]
        .value_counts()
        .head(15)
        .reset_index()
    )

    crime_counts.columns = [
        "Crime Type",
        "Geographic Cells"
    ]

    st.subheader(
        "Dominant Crime Type"
    )

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    sns.barplot(
        data=crime_counts,
        x="Geographic Cells",
        y="Crime Type",
        ax=ax
    )

    ax.set_title(
        "Dominant Crime Type Across Geographic Cells"
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # --------------------------------------------------------
    # Severity distribution
    # --------------------------------------------------------

    st.subheader(
        "Average Severity Distribution"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.histplot(
        geo_profile["Avg_Severity"],
        bins=20,
        kde=True,
        ax=ax
    )

    ax.set_xlabel(
        "Average Severity"
    )

    ax.set_ylabel(
        "Geographic Cells"
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # --------------------------------------------------------
    # Arrest vs domestic
    # --------------------------------------------------------

    st.subheader(
        "Arrest Rate vs Domestic Incident Rate"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.scatterplot(
        data=geo_profile,
        x="Arrest_Rate",
        y="Domestic_Rate",
        size="Total_Crimes",
        alpha=0.5,
        ax=ax,
        legend=False
    )

    ax.set_xlabel(
        "Arrest Rate"
    )

    ax.set_ylabel(
        "Domestic Rate"
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)


# ============================================================
# PAGE 3 — GEOGRAPHIC ANALYSIS
# ============================================================

elif page == "Geographic Analysis":

    st.header("Geographic Analysis")

    st.markdown(
        """
        Explore aggregate geographic cells created from rounded
        latitude and longitude coordinates.
        """
    )

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    min_crimes = st.slider(
        "Minimum reported incidents in cell",
        min_value=1,
        max_value=int(
            geo_profile["Total_Crimes"].quantile(0.99)
        ),
        value=5
    )

    filtered_geo = geo_profile[
        geo_profile["Total_Crimes"] >= min_crimes
    ].copy()

    st.metric(
        "Cells matching filter",
        f"{len(filtered_geo):,}"
    )

    # --------------------------------------------------------
    # Top cells
    # --------------------------------------------------------

    st.subheader(
        "Highest-Volume Geographic Cells"
    )

    top_cells = (
        filtered_geo
        .sort_values(
            "Total_Crimes",
            ascending=False
        )
        .head(20)
    )

    display_columns = [
        "Geo_Cell",
        "Latitude",
        "Longitude",
        "Total_Crimes",
        "Arrest_Rate",
        "Domestic_Rate",
        "Avg_Severity",
        "Dominant_Crime"
    ]

    st.dataframe(
        top_cells[display_columns],
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    st.subheader(
        "Geographic Distribution"
    )

    map_data = filtered_geo[
        [
            "Latitude",
            "Longitude"
        ]
    ].dropna()

    st.map(
        map_data,
        latitude="Latitude",
        longitude="Longitude",
        size=8
    )


# ============================================================
# PAGE 4 — CLUSTERING
# ============================================================

elif page == "Clustering":

    st.header("Clustering Analysis")

    st.markdown(
        """
        PatrolIQ evaluates multiple unsupervised-learning approaches
        to identify different patterns among geographic cells.
        """
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    st.subheader(
        "Clustering Model Comparison"
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Higher silhouette indicates stronger separation. "
        "Lower Davies-Bouldin indicates better compactness/separation. "
        "These metrics are interpreted together with cluster size "
        "and pattern interpretability."
    )

    st.divider()

    # --------------------------------------------------------
    # K-Means
    # --------------------------------------------------------

    st.subheader(
        "K-Means — K=8"
    )

    col1, col2, col3 = st.columns(3)

    kmeans_counts = (
        geo_model["KMeans_Cluster"]
        .value_counts()
        .sort_index()
    )

    with col1:
        st.metric(
            "Clusters",
            "8"
        )

    with col2:
        st.metric(
            "Silhouette",
            "0.0390"
        )

    with col3:
        st.metric(
            "Davies-Bouldin",
            "2.4327"
        )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        x=kmeans_counts.index,
        y=kmeans_counts.values,
        ax=ax
    )

    ax.set_xlabel(
        "K-Means Cluster"
    )

    ax.set_ylabel(
        "Geographic Cells"
    )

    ax.set_title(
        "K-Means Cluster Distribution"
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    st.subheader(
        "K-Means Cluster Profiles"
    )

    st.dataframe(
        kmeans_profile,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # DBSCAN
    # --------------------------------------------------------

    st.subheader(
        "DBSCAN — eps=7.0"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Clusters",
            "2"
        )

    with col2:
        st.metric(
            "Silhouette",
            "0.4178"
        )

    with col3:
        st.metric(
            "Noise",
            "2.11%"
        )

    st.dataframe(
        dbscan_profile,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        """
        DBSCAN classified 493 geographic cells as noise.
        The remaining structure was dominated by one large cluster,
        with a very small secondary cluster. Therefore, DBSCAN is
        treated primarily as a density/outlier analysis rather than
        a balanced geographic segmentation.
        """
    )


# ============================================================
# PAGE 5 — DIMENSIONALITY REDUCTION
# ============================================================

elif page == "Dimensionality Reduction":

    st.header(
        "Dimensionality Reduction"
    )

    st.markdown(
        """
        PCA summarizes the 50-dimensional geographic feature space,
        while t-SNE provides a nonlinear two-dimensional visualization
        of local structure.
        """
    )

    # --------------------------------------------------------
    # PCA metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Original Features",
            "50"
        )

    with col2:
        st.metric(
            "70% Variance",
            "27 PCs"
        )

    with col3:
        st.metric(
            "90% Variance",
            "37 PCs"
        )

    with col4:
        st.metric(
            "PC1 + PC2",
            "11.07%"
        )

    st.divider()

    # --------------------------------------------------------
    # PCA variance
    # --------------------------------------------------------

    st.subheader(
        "PCA Explained Variance"
    )

    pca_variance = (
        pca.explained_variance_ratio_
    )

    cumulative_variance = np.cumsum(
        pca_variance
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        range(
            1,
            len(cumulative_variance) + 1
        ),
        cumulative_variance
    )

    ax.axhline(
        0.70,
        linestyle="--",
        label="70%"
    )

    ax.axhline(
        0.80,
        linestyle="--",
        label="80%"
    )

    ax.axhline(
        0.90,
        linestyle="--",
        label="90%"
    )

    ax.axhline(
        0.95,
        linestyle="--",
        label="95%"
    )

    ax.set_xlabel(
        "Principal Components"
    )

    ax.set_ylabel(
        "Cumulative Explained Variance"
    )

    ax.set_title(
        "PCA Cumulative Explained Variance"
    )

    ax.legend()

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # --------------------------------------------------------
    # Existing PCA / t-SNE images
    # --------------------------------------------------------

    st.subheader(
        "PCA 2D Projection"
    )

    pca_image = os.path.join(
        VISUALIZATION_DIR,
        "pca_2d_kmeans.png"
    )

    if os.path.exists(pca_image):
        st.image(pca_image, use_container_width=True)
    else:
        st.warning(f"PCA image not found: {pca_image}")

    st.subheader(
        "t-SNE Projection"
    )

    tsne_image = os.path.join(
        VISUALIZATION_DIR,
        "tsne_kmeans.png"
    )

    if os.path.exists(tsne_image):
        st.image(tsne_image, use_container_width=True)
    else:
        st.warning(f"t-SNE image not found: {tsne_image}")


# ============================================================
# PAGE 6 — MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.header(
        "Model Evaluation"
    )

    st.markdown(
        """
        Quantitative clustering metrics are presented alongside
        cluster-size characteristics to avoid interpreting an
        internal metric in isolation.
        """
    )

    # --------------------------------------------------------
    # Comparison table
    # --------------------------------------------------------

    st.subheader(
        "Clustering Evaluation"
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # Silhouette comparison
    # --------------------------------------------------------

    st.subheader(
        "Silhouette Score"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=comparison,
        x="Algorithm",
        y="Silhouette_Score",
        ax=ax
    )

    ax.set_ylabel(
        "Silhouette Score"
    )

    ax.set_xlabel(
        "Algorithm"
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # --------------------------------------------------------
    # Davies-Bouldin comparison
    # --------------------------------------------------------

    st.subheader(
        "Davies-Bouldin Index"
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=comparison,
        x="Algorithm",
        y="Davies_Bouldin_Index",
        ax=ax
    )

    ax.set_ylabel(
        "Davies-Bouldin Index"
    )

    ax.set_xlabel(
        "Algorithm"
    )

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)

    # --------------------------------------------------------
    # Methodology note
    # --------------------------------------------------------

    st.warning(
        """
        Internal clustering metrics should not be interpreted alone.
        For PatrolIQ, cluster size, noise proportion and interpretability
        are considered alongside silhouette and Davies-Bouldin metrics.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "PatrolIQ • Aggregate reported-crime analytics"
)
