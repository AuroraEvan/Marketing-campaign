import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("Customer Analysis and Marketing Campaign Optimization (P7)")

st.write("""
**P7** implements the data science pipeline from **P6** to optimize marketing campaigns, segment customers, 
and improve inventory management for ABC Manufacturing, using the **marketing_campaign.csv** dataset. 
This app performs the following steps: data collection, preprocessing, analysis, visualization (5 charts), 
model training (KMeans, Linear Regression), and evaluation (R², MSE, MAE, Silhouette Score).
""")

# Step 1: Data Collection
st.header("Step 1: Data Collection")
url = "https://raw.githubusercontent.com/AuroraEvan/Marketing-campaign.github.io/main/marketing_campaign.csv"
data = pd.read_csv(url)
st.write(data.head())

try:
    # Step 2: Data Preprocessing
    st.header("Step 2: Data Preprocessing")

    # Handle nulls and duplicates
    data['Income'].fillna(data['Income'].mean(), inplace=True)
    data = data.drop_duplicates()
    st.write("Number of records after removing nulls and duplicates:", len(data))

    # Handle outliers
    data = data[data['Income'] < 200000]
    st.write("Number of records after removing outliers:", len(data))
    st.write("Average Income:", data['Income'].mean())

    # Convert Dt_Customer and create new columns
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
    data['Years_Since_Customer'] = (pd.Timestamp('2025-07-30') - data['Dt_Customer']).dt.days / 365.25
    data['Total_Spend'] = (data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] +
                           data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds'])
    data['Campaign_Acceptance'] = data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                                        'AcceptedCmp4', 'AcceptedCmp5', 'Response']].sum(axis=1)
    st.write("Years since customer registration (min-max):",
             data['Years_Since_Customer'].min(), "-", data['Years_Since_Customer'].max())
    st.write("Average Total Spend:", data['Total_Spend'].mean())
    st.write("Average Campaign Acceptance:", data['Campaign_Acceptance'].mean())

    # Standardize data
    features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
                'NumStorePurchases', 'NumWebVisitsMonth', 'Years_Since_Customer']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    st.write("Standardized data (sample):", data_scaled[:2])

    # Step 3: Data Analysis
    st.header("Step 3: Data Analysis")
    st.write("**Income Statistics:**")
    st.write(data['Income'].describe())
    st.write("**Total Spend Statistics:**")
    st.write(data['Total_Spend'].describe())
    st.write("**Campaign Acceptance Rate:**")
    st.write(data['Campaign_Acceptance'].value_counts(normalize=True))

    # Step 4: Data Visualization
    st.header("Step 4: Data Visualization")

    # Chart 1 - Scatter Plot (Income vs Total Spend)
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_scaled)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=data['Income'], y=data['Total_Spend'], hue=data['Cluster'],
                    size=data['Campaign_Acceptance'], sizes=(20, 200), palette='viridis', ax=ax)
    plt.title('Customer Segmentation: Income vs Total Spend')
    plt.xlabel('Income (USD)')
    plt.ylabel('Total Spend (USD)')
    st.pyplot(fig)

    # Chart 2 - Boxplot (Income by Cluster)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Income', data=data, ax=ax)
    plt.title('Income Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Income (USD)')
    st.pyplot(fig)

    # Chart 3 - Stacked Bar Plot (Spending by Product)
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                     'MntSweetProducts', 'MntGoldProds']
    spending_by_cluster = data.groupby('Cluster')[spending_cols].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    spending_by_cluster.plot(kind='bar', stacked=True, ax=ax)
    plt.title('Average Spending by Product and Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Average Spending (USD)')
    plt.legend(title='Product')
    st.pyplot(fig)

    # Chart 4 - Heatmap (Correlation)
    corr_cols = ['Income', 'Recency', 'Total_Spend', 'NumWebPurchases', 'NumStorePurchases']
    corr_matrix = data[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title('Correlation Heatmap')
    st.pyplot(fig)

    # Chart 5 - Line Plot (Recency vs Total Spend)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in range(4):
        cluster_data = data[data['Cluster'] == cluster]
        ax.plot(cluster_data['Recency'], cluster_data['Total_Spend'], label=f'Cluster {cluster}')
    plt.title('Recency vs Total Spend by Cluster')
    plt.xlabel('Recency (Days)')
    plt.ylabel('Total Spend (USD)')
    plt.legend()
    st.pyplot(fig)

    # Step 5: Model Training
    st.header("Step 5: Model Training")
    st.write("KMeans clustering completed, 4 clusters, labels stored in Cluster column")

    # Linear Regression
    X = data[features]
    y = data['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    st.write("Linear Regression model trained.")
    st.write("Coefficients:", lr_model.coef_)

    # Step 6: Model Evaluation
    st.header("Step 6: Model Evaluation")
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write("**Linear Regression Evaluation:**")
    st.write(f"R² Score: {r2:.4f}")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

    sil_score = silhouette_score(data_scaled, data['Cluster'])
    st.write("Silhouette Score for KMeans:", sil_score)

except Exception as e:
    st.error(f"An error occurred: {str(e)}. Please check the file format or content.")
