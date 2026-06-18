import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 1. 讀取階段一產出的資料表
df = pd.read_csv(r"C:\Users\user\Desktop\model_features.csv")

# 2. 缺失值處理 (Imputation)
# 10月從未購買過的人，Monetary、purchase_count、轉換率會是 NaN，需自動補 0
df['monetary'] = df['monetary'].fillna(0)
df['purchase_count'] = df['purchase_count'].fillna(0)
df['cart_to_purchase_rate'] = df['cart_to_purchase_rate'].fillna(0)

# [選填額外步驟] 如果你在階段一有撈取類別欄位（例如最常買的類別），可用 One-Hot Encoding：
# df = pd.get_dummies(df, columns=['most_bought_category'], drop_first=True)

# 3. 提取基礎特徵欄位（排除 user_id 與答案 Is_Churn）
features = [
    'recency', 'frequency', 'monetary', 
    'view_count', 'cart_count', 'purchase_count',
    'cart_to_purchase_rate', 'active_days'
]
X_raw = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 4. K-Means 特徵工程（全量分群）
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 4. [關鍵診斷] 檢查每一群的「個性」
# 將中心點還原成原始數值，以便閱讀 (例如：元、次、天)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_analysis = pd.DataFrame(centers, columns=features)

print(f"--- 分群結果 (K={k}) ---")
print(cluster_analysis.round(2)) # 取到小數點後兩位，比較好閱讀

# 5. 觀察該群組的用戶佔比 (看看有沒有哪一群人太少，變成孤島)
print("\n--- 各群組用戶比例 ---")
print(df['Cluster'].value_counts(normalize=True))