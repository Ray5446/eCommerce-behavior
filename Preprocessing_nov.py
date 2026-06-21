import pandas as pd
import joblib

# 1. 讀取你剛剛產出的 11 月特徵表
df = pd.read_csv(r"C:\Users\user\Desktop\model_features_nov.csv")

df['monetary'] = df['monetary'].fillna(0)
df['purchase_count'] = df['purchase_count'].fillna(0)
df['cart_to_purchase_rate'] = df['cart_to_purchase_rate'].fillna(0)

# 2. 載入當初訓練 10 月資料時存下的模型物件
# 確保這些 .pkl 檔案跟這份程式碼在同一個資料夾，或者使用完整路徑
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans.pkl')

# 3. 定義當初訓練用的特徵欄位（務必與 10 月訓練時一模一樣）
features = [
    'recency', 'frequency', 'monetary', 
    'view_count', 'cart_count', 'purchase_count',
    'cart_to_purchase_rate', 'active_days'
]

# 4. 數據標準化 (使用 10 月的 scaler)
# 這裡使用 transform 而非 fit_transform
X_raw = df[features]
X_scaled = scaler.transform(X_raw)

# 5. 套用分群模型 (使用 10 月的 kmeans)
# 這裡直接 predict，把 11 月用戶歸類到 10 月學到的 4 個群組中
df['Cluster'] = kmeans.predict(X_scaled)

# 6. 產出結果並存檔
df.to_csv(r"C:\Users\user\Desktop\model_features_with_cluster_nov.csv", index=False)

centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_analysis = pd.DataFrame(centers, columns=features)

print(f"--- 分群結果 (K={4}) ---")
print(cluster_analysis.round(2)) # 取到小數點後兩位，比較好閱讀

# 5. 觀察該群組的用戶佔比 (看看有沒有哪一群人太少，變成孤島)
print("\n--- 各群組用戶比例 ---")
print(df['Cluster'].value_counts(normalize=True))