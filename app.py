# streamlit_customer_clustering_app.py
# Streamlit app for EDA, clustering comparison, model export (kmeans_model.pkl)
# Save this file and run: streamlit run streamlit_customer_clustering_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

sns.set()

st.set_page_config(page_title='Customer Clustering Explorer', layout='wide')
st.title('Customer Clustering Explorer')
st.markdown("""
This app performs exploratory data analysis (EDA) on a customers dataset, compares clustering algorithms
(KMeans, Hierarchical/Agglomerative, DBSCAN), evaluates them with cluster metrics, visualizes results (PCA),
and exports a trained KMeans model as `kmeans_model.pkl` for use in production / Streamlit apps.
""")

# --- Sidebar: Data upload / options ---
st.sidebar.header('Data / Settings')
uploaded_file = st.sidebar.file_uploader('Upload CSV file (Mall_Customers.csv format suggested)', type=['csv'])
use_sample = st.sidebar.checkbox('Use bundled sample dataset', value=True)

if uploaded_file is None and not use_sample:
    st.sidebar.warning('Please upload a CSV or enable the bundled sample dataset.')

# Provide a small sample dataset if requested
@st.cache_data
def load_sample():
    data = {
        'CustomerID': list(range(1,201)),
        'Gender': np.random.choice(['Male','Female'], size=200),
        'Age': np.random.randint(18,70,size=200),
        'Annual Income (k$)': np.random.randint(15,140,size=200),
        'Spending Score (1-100)': np.random.randint(1,101,size=200)
    }
    return pd.DataFrame(data)

def load_data():
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error('Failed to read uploaded CSV: {}'.format(e))
            return None
    if use_sample:
        return load_sample()
    return None

customer_data = load_data()
if customer_data is None:
    st.stop()

# show raw data
with st.expander('Show raw data'):
    st.dataframe(customer_data.head(200))

# Basic info
st.subheader('Dataset Summary')
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.metric('Rows', customer_data.shape[0])
with col2:
    st.metric('Columns', customer_data.shape[1])
with col3:
    st.metric('Missing values', int(customer_data.isnull().sum().sum()))

st.write('---')

# --- Extended EDA ---
st.subheader('Exploratory Data Analysis (EDA)')
eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.write('### Numerical Summary')
    st.write(customer_data.describe())

    st.write('### Gender Distribution')
    if 'Gender' in customer_data.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='Gender', data=customer_data, ax=ax)
        ax.set_title('Gender Count')
        st.pyplot(fig)
    else:
        st.info('No Gender column found in dataset.')

with eda_col2:
    st.write('### Correlation Heatmap (numerical cols)')
    num = customer_data.select_dtypes(include=np.number)
    fig2, ax2 = plt.subplots(figsize=(6,4))
    sns.heatmap(num.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

# Distributions
st.write('### Distributions')
dist_col1, dist_col2 = st.columns(2)
with dist_col1:
    if 'Annual Income (k$)' in customer_data.columns:
        fig3, ax3 = plt.subplots()
        sns.histplot(customer_data['Annual Income (k$)'], bins=20, kde=True, ax=ax3)
        ax3.set_title('Annual Income (k$)')
        st.pyplot(fig3)
with dist_col2:
    if 'Spending Score (1-100)' in customer_data.columns:
        fig4, ax4 = plt.subplots()
        sns.histplot(customer_data['Spending Score (1-100)'], bins=20, kde=True, ax=ax4)
        ax4.set_title('Spending Score (1-100)')
        st.pyplot(fig4)

# Pairplot (small sample for performance)
if customer_data.shape[0] <= 1000:
    st.write('### Pairplot: Age, Annual Income, Spending Score')
    cols = [c for c in customer_data.columns if c in ['Age','Annual Income (k$)','Spending Score (1-100)']]
    if len(cols) == 3:
        pair_fig = sns.pairplot(customer_data[cols])
        st.pyplot(pair_fig)

st.write('---')

# --- Prepare features for clustering ---
st.subheader('Clustering: features selection')
feature_cols = st.multiselect(
    'Select two numerical features for clustering (default: Annual Income, Spending Score)',
    options=[c for c in customer_data.columns if np.issubdtype(customer_data[c].dtype, np.number)],
    default=['Annual Income (k$)','Spending Score (1-100)']
)

if len(feature_cols) != 2:
    st.warning('Please select exactly two numeric features to continue.')
    st.stop()

X = customer_data[feature_cols].values

# Option: scale features
scale = st.checkbox('Scale features (StandardScaler)', value=False)
if scale:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    scaler = None
    X_scaled = X

# --- Elbow method for KMeans ---
st.subheader('Elbow method (WCSS)')
max_k = st.slider('Max clusters to test', min_value=3, max_value=12, value=10)
wcss = []
for k in range(1, max_k+1):
    km = KMeans(n_clusters=k, init='k-means++', random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

fig_elb, ax_elb = plt.subplots()
ax_elb.plot(range(1, max_k+1), wcss, marker='o')
ax_elb.set_xlabel('Number of clusters')
ax_elb.set_ylabel('WCSS')
ax_elb.set_title('Elbow plot')
st.pyplot(fig_elb)

# Let user choose n_clusters
n_clusters = st.number_input('Choose number of clusters for KMeans and Agglomerative',
                             min_value=2, max_value=12, value=5, step=1)

# --- Train and compare clustering algorithms ---
st.subheader('Train & Compare Clustering Algorithms')
train_button = st.button('Run clustering and evaluate')
if train_button:
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    y_km = kmeans.fit_predict(X_scaled)

    # Agglomerative
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    y_hc = hc.fit_predict(X_scaled)

    # DBSCAN
    eps = st.sidebar.slider('DBSCAN eps (distance)', 0.5, 30.0, 5.0)
    min_s = st.sidebar.slider('DBSCAN min_samples', 2, 20, 5)
    dbs = DBSCAN(eps=eps, min_samples=min_s)
    y_db = dbs.fit_predict(X_scaled)

    # Evaluation metrics
    st.write('### Clustering Metrics')
    try:
        km_sil = silhouette_score(X_scaled, y_km)
    except Exception:
        km_sil = np.nan
    try:
        km_db = davies_bouldin_score(X_scaled, y_km)
    except Exception:
        km_db = np.nan
    try:
        km_ch = calinski_harabasz_score(X_scaled, y_km)
    except Exception:
        km_ch = np.nan

    try:
        hc_sil = silhouette_score(X_scaled, y_hc)
    except Exception:
        hc_sil = np.nan
    try:
        db_sil = silhouette_score(X_scaled[y_db!=-1], y_db[y_db!=-1]) if len(np.unique(y_db))>1 else np.nan
    except Exception:
        db_sil = np.nan

    metrics_df = pd.DataFrame({
        'Model': ['KMeans','Agglomerative','DBSCAN'],
        'Silhouette': [km_sil, hc_sil, db_sil],
        'Davies-Bouldin (lower better)': [km_db, np.nan, np.nan],
        'Calinski-Harabasz (higher better)': [km_ch, np.nan, np.nan]
    })
    st.dataframe(metrics_df)

    # Visualizations
    vis_col1, vis_col2 = st.columns(2)
    with vis_col1:
        st.write('KMeans clusters')
        fig_k, axk = plt.subplots(figsize=(6,5))
        scatter = axk.scatter(X[:,0], X[:,1], c=y_km, cmap='tab10', s=50)
        axk.scatter(
            kmeans.cluster_centers_[:,0] if not scale else scaler.inverse_transform(kmeans.cluster_centers_)[:,0],
            kmeans.cluster_centers_[:,1] if not scale else scaler.inverse_transform(kmeans.cluster_centers_)[:,1],
            s=200, marker='X', label='Centroids'
        )
        axk.set_xlabel(feature_cols[0])
        axk.set_ylabel(feature_cols[1])
        axk.set_title('KMeans (k={})'.format(n_clusters))
        axk.legend()
        st.pyplot(fig_k)

    with vis_col2:
        st.write('Agglomerative clusters')
        fig_h, axh = plt.subplots(figsize=(6,5))
        axh.scatter(X[:,0], X[:,1], c=y_hc, cmap='tab10', s=50)
        axh.set_xlabel(feature_cols[0])
        axh.set_ylabel(feature_cols[1])
        axh.set_title('Agglomerative (k={})'.format(n_clusters))
        st.pyplot(fig_h)

    st.write('DBSCAN clusters (label -1 = noise)')
    fig_d, axd = plt.subplots(figsize=(6,5))
    axd.scatter(X[:,0], X[:,1], c=y_db, cmap='tab10', s=50)
    axd.set_xlabel(feature_cols[0])
    axd.set_ylabel(feature_cols[1])
    st.pyplot(fig_d)

    st.write('### PCA projection of data and KMeans clusters')
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig_p, axp = plt.subplots()
    axp.scatter(X_pca[:,0], X_pca[:,1], c=y_km, s=40)
    axp.set_title('PCA (2 components) colored by KMeans cluster')
    st.pyplot(fig_p)

    # --- Cluster Profiling ---
    st.subheader("Cluster Profiling (KMeans)")
    customer_data['Cluster'] = y_km
    profile = customer_data.groupby('Cluster').agg({
        feature_cols[0]: ['mean','min','max'],
        feature_cols[1]: ['mean','min','max']
    })

    # Assign human-readable labels
    def label_cluster(row):
        mean_x = row[(feature_cols[0], 'mean')]
        mean_y = row[(feature_cols[1], 'mean')]
        if mean_x > np.percentile(customer_data[feature_cols[0]], 66) and mean_y > np.percentile(customer_data[feature_cols[1]], 66):
            return "High Income, High Spending"
        elif mean_x < np.percentile(customer_data[feature_cols[0]], 33) and mean_y < np.percentile(customer_data[feature_cols[1]], 33):
            return "Low Income, Low Spending"
        elif mean_x >= np.percentile(customer_data[feature_cols[0]], 33) and mean_x <= np.percentile(customer_data[feature_cols[0]], 66):
            return "Mid Income, Mid Spending"
        else:
            return "Other Segment"

    profile['Segment'] = profile.apply(label_cluster, axis=1)
    st.dataframe(profile)

    # Save model + scaler + features + profile
    save_path = 'kmeans_model.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump({'model': kmeans,
                     'scaler': scaler,
                     'features': feature_cols,
                     'profile': profile}, f)

    st.success(f'KMeans model + profiling saved to `{save_path}`')
    with open(save_path, 'rb') as f:
        btn = st.download_button(label='Download kmeans_model.pkl',
                                 data=f, file_name='kmeans_model.pkl',
                                 mime='application/octet-stream')

# --- Prediction / Inference UI ---
st.write('---')
st.subheader('Predict cluster for a new customer')
with st.form('predict_form'):
    val1 = st.number_input(f'{feature_cols[0]}', value=float(customer_data[feature_cols[0]].median()))
    val2 = st.number_input(f'{feature_cols[1]}', value=float(customer_data[feature_cols[1]].median()))
    submitted = st.form_submit_button('Predict (uses saved kmeans_model.pkl if available)')

if submitted:
    try:
        with open('kmeans_model.pkl', 'rb') as f:
            saved = pickle.load(f)
            model = saved['model']
            saved_scaler = saved['scaler']
            saved_features = saved['features']
            saved_profile = saved['profile']
    except Exception:
        st.error('Trained kmeans_model.pkl not found. Run training above to create it.')
        model = None
        saved_scaler = None
        saved_profile = None

    if model is not None:
        arr = np.array([[val1, val2]])
        if saved_scaler is not None:
            arr = saved_scaler.transform(arr)
        pred = model.predict(arr)[0]
        st.write(f'Predicted cluster: **{int(pred)}**')

        if saved_profile is not None and 'Segment' in saved_profile.columns:
            seg_name = saved_profile.loc[pred, 'Segment']
            st.write(f'→ Segment: **{seg_name}**')

st.write('---')
st.write('Notes:')
st.write('- The app uses two features for clustering by default. You can change the features in the sidebar.')
st.write('- If your dataset columns have different names, choose the appropriate columns via the multiselect.')
st.write('- For production, adapt file paths and consider storing models in cloud storage (S3, GCS) or a model registry.')
