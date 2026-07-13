import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Customer Segmentation System", layout="wide")

st.title("🛍 Customer Segmentation System")
st.write("Upload a customer dataset to identify customer segments using K-Means Clustering.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Information")
    st.write(df.describe(include='all'))

    data = df.copy()

    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    st.subheader("Select Features for Clustering")

    features = st.multiselect(
        "Choose Features",
        data.columns,
        default=data.columns[:2]
    )

    if len(features) >= 2:

        X = data[features]

        clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=4
        )

        model = KMeans(n_clusters=clusters, random_state=42, n_init=10)

        data["Cluster"] = model.fit_predict(X)

        st.success("Customer Segmentation Completed Successfully!")

        st.subheader("Segmented Dataset")
        st.dataframe(data)

        fig = px.scatter(
            data,
            x=features[0],
            y=features[1],
            color=data["Cluster"].astype(str),
            title="Customer Segments",
            hover_data=data.columns
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Customers in Each Segment")
        st.bar_chart(data["Cluster"].value_counts())

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Segmented Dataset",
            csv,
            "customer_segments.csv",
            "text/csv"
        )

    else:
        st.warning("Please select at least two features.")
