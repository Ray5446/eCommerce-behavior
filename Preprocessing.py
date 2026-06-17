import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. 讀取階段一產出的資料表
df = pd.read_csv('churn_training_dataset.csv')

# 2. 缺失值處理 (Imputation)
# 10月從未購買過的人，Monetary、purchase_count、轉換率會是 NaN，需自動補 0
df['Monetary'] = df['Monetary'].fillna(0)
df['purchase_count'] = df['purchase_count'].fillna(0)
df['cart_to_purchase_rate'] = df['cart_to_purchase_rate'].fillna(0)

# [選填額外步驟] 如果你在階段一有撈取類別欄位（例如最常買的類別），可用 One-Hot Encoding：
# df = pd.get_dummies(df, columns=['most_bought_category'], drop_first=True)

# 3. 提取基礎特徵欄位（排除 user_id 與答案 Is_Churn）
feature_cols = [
    'Recency', 'Frequency', 'Monetary', 
    'view_count', 'cart_count', 'remove_count', 'purchase_count',
    'cart_to_purchase_rate', 'active_days'
]
X_raw = df[feature_cols]

# 4. K-Means 特徵工程（全量分群）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("✅ 步驟一完成：K-Means 分群特徵已成功加入資料表！")