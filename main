st.write("""
**P7** triển khai quy trình khoa học dữ liệu từ **P6** để tối ưu hóa chiến dịch tiếp thị, phân khúc khách hàng, 
và cải thiện quản lý kho cho ABC Manufacturing, sử dụng tập dữ liệu **marketing_campaign.csv**. 
Ứng dụng này thực hiện các bước: thu thập dữ liệu, tiền xử lý, phân tích, trực quan hóa (5 biểu đồ), 
huấn luyện mô hình (KMeans, Linear Regression), và đánh giá (R², MSE, MAE, Silhouette Score).
""")

# Bước 1: Thu thập dữ liệu
st.header("Bước 1: Thu thập dữ liệu")
st.write("Tải file **marketing_campaign.csv** từ Kaggle (phân cách bằng tab).")
uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])

if uploaded_file is not None:
    # Đọc dữ liệu
    data = pd.read_csv(uploaded_file, sep='\t')
    st.write("**Kích thước dữ liệu:**", data.shape)
    st.write("**Thông tin cột:**")
    st.write(data.info())
    st.write("**Mẫu dữ liệu:**")
    st.dataframe(data.head())

    # Bước 2: Tiền xử lý dữ liệu
    st.header("Bước 2: Tiền xử lý dữ liệu")
    
    # Code 2.1: Xử lý null và trùng lặp
    data['Income'].fillna(data['Income'].mean(), inplace=True)
    data = data.drop_duplicates()
    st.write("Số bản ghi sau xóa null và trùng lặp:", len(data))

    # Code 2.2: Xử lý outlier
    data = data[data['Income'] < 200000]
    st.write("Số bản ghi sau xóa outlier:", len(data))
    st.write("Thu nhập trung bình:", data['Income'].mean())

    # Code 2.3: Chuyển đổi Dt_Customer và tạo cột mới
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')
    data['Years_Since_Customer'] = (pd.Timestamp('2025-07-30') - data['Dt_Customer']).dt.days / 365.25
    data['Total_Spend'] = (data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] +
                           data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds'])
    data['Campaign_Acceptance'] = data[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                                        'AcceptedCmp4', 'AcceptedCmp5', 'Response']].sum(axis=1)
    st.write("Số năm kể từ đăng ký (min-max):", data['Years_Since_Customer'].min(), "-", data['Years_Since_Customer'].max())
    st.write("Total_Spend trung bình:", data['Total_Spend'].mean())
    st.write("Campaign_Acceptance trung bình:", data['Campaign_Acceptance'].mean())

    # Code 2.4: Chuẩn hóa dữ liệu
    features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 
                'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
                'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
                'NumStorePurchases', 'NumWebVisitsMonth', 'Years_Since_Customer']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    st.write("Dữ liệu chuẩn hóa (mẫu):", data_scaled[:2])

    # Bước 3: Phân tích dữ liệu
    st.header("Bước 3: Phân tích dữ liệu")
    
    # Code 3.1: Thống kê cơ bản
    st.write("**Thống kê thu nhập:**")
    st.write(data['Income'].describe())
    st.write("**Thống kê chi tiêu tổng:**")
    st.write(data['Total_Spend'].describe())
    st.write("**Tỷ lệ chấp nhận chiến dịch:**")
    st.write(data['Campaign_Acceptance'].value_counts(normalize=True))

    # Bước 4: Trực quan hóa dữ liệu
    st.header("Bước 4: Trực quan hóa dữ liệu")

    # Code 4.1: Biểu đồ 1 - Scatter Plot (Income vs Total_Spend)
    kmeans = KMeans(n_clusters=4, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data_scaled)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=data['Income'], y=data['Total_Spend'], hue=data['Cluster'], 
                    size=data['Campaign_Acceptance'], sizes=(20, 200), palette='viridis', ax=ax)
    plt.title('Phân khúc khách hàng: Thu nhập vs Chi tiêu tổng')
    plt.xlabel('Thu nhập (USD)')
    plt.ylabel('Chi tiêu tổng (USD)')
    st.pyplot(fig)

    # Code 4.2: Biểu đồ 2 - Boxplot (Income by Cluster)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Income', data=data, ax=ax)
    plt.title('Phân phối thu nhập theo cụm')
    plt.xlabel('Cụm')
    plt.ylabel('Thu nhập (USD)')
    st.pyplot(fig)

    # Code 4.3: Biểu đồ 3 - Stacked Bar Plot (Spending by Product)
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
                     'MntSweetProducts', 'MntGoldProds']
    spending_by_cluster = data.groupby('Cluster')[spending_cols].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    spending_by_cluster.plot(kind='bar', stacked=True, ax=ax)
    plt.title('Chi tiêu trung bình theo sản phẩm và cụm')
    plt.xlabel('Cụm')
    plt.ylabel('Chi tiêu trung bình (USD)')
    plt.legend(title='Sản phẩm')
    st.pyplot(fig)

    # Code 4.4: Biểu đồ 4 - Heatmap (Correlation)
    corr_cols = ['Income', 'Recency', 'Total_Spend', 'NumWebPurchases', 'NumStorePurchases']
    corr_matrix = data[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title('Biểu đồ nhiệt tương quan')
    st.pyplot(fig)

    # Code 4.5: Biểu đồ 5 - Line Plot (Recency vs Total_Spend)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in range(4):
        cluster_data = data[data['Cluster'] == cluster]
        ax.plot(cluster_data['Recency'], cluster_data['Total_Spend'], label=f'Cụm {cluster}')
    plt.title('Recency vs Chi tiêu tổng theo cụm')
    plt.xlabel('Recency (Ngày)')
    plt.ylabel('Chi tiêu tổng (USD)')
    plt.legend()
    st.pyplot(fig)

    # Bước 5: Huấn luyện mô hình
    st.header("Bước 5: Huấn luyện mô hình")
    
    # Code 5.1: Huấn luyện KMeans
    st.write("Đã phân cụm với KMeans, 4 cụm, nhãn lưu trong cột Cluster")

    # Code 5.2: Huấn luyện Linear Regression
    X = data[features]
    y = data['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    st.write("Mô hình Linear Regression đã huấn luyện.")
    st.write("Hệ số:", lr_model.coef_)

    # Bước 6: Đánh giá mô hình
    st.header("Bước 6: Đánh giá mô hình")

    # Code 6.1: Đánh giá Linear Regression
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    st.write("**Đánh giá mô hình Linear Regression:**")
    st.write(f"R² Score: {r2:.4f}")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Code 6.2: Đánh giá KMeans
    sil_score = silhouette_score(data_scaled, data['Cluster'])
    st.write("Silhouette Score cho KMeans:", sil_score)

else:
    st.warning("Vui lòng tải file CSV để tiếp tục.")
