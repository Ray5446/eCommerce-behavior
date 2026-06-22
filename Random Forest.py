import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# 1. 讀取資料
# 建議將路徑統一管理，方便後續修改
path_oct = r"C:\Users\user\Desktop\model_features_with_cluster_oct.csv"
path_nov = r"C:\Users\user\Desktop\model_features_with_cluster_nov.csv"

train_df = pd.read_csv(path_oct)
nov_df = pd.read_csv(path_nov)

# 2. 定義特徵欄位
features = [
    'recency', 'frequency', 'monetary', 'active_days', 
    'view_count', 'cart_count', 'cart_to_purchase_rate'
]

# 3. 訓練模型
X_train = train_df[features]
y_train = train_df['label_has_purchased_nov']

# 1. 強制只取 5% 資料來做 GridSearch (約 15 萬筆，非常足夠找出最佳權重)
sample_df = train_df.sample(frac=0.05, random_state=42)
X_sample = sample_df[features]
y_sample = sample_df['label_has_purchased_nov']

# 2. 進行參數搜尋
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(n_estimators=50, random_state=42), # 減少樹的數量
    param_grid={'class_weight': [{0: 1, 1: v} for v in [1, 3, 5]]},
    scoring='f1',
    cv=2, # 減少交叉驗證次數
    n_jobs=-1
)

grid_search.fit(X_sample, y_sample)
print(f"✅ 找到最佳權重參數: {grid_search.best_params_}")

# 3. 拿到最佳權重後，再用「全量資料」訓練最終模型
# 因為不需要 GridSearch，這步驟會快非常多！
best_params = grid_search.best_params_
rf = RandomForestClassifier(n_estimators=100, random_state=42, **best_params)
rf.fit(X_train, y_train)

joblib.dump(rf, 'rf_model_optimized.pkl')
print("✅ 最終模型訓練完成並已儲存！")

# 4. 預測 11 月數據 (預測 12 月流失機率)
X_nov = nov_df[features]
nov_df['Purchase_Probability'] = rf.predict_proba(X_nov)[:, 1]
nov_df.to_csv(r"C:\Users\user\Desktop\Churn_Prediction_List_Nov.csv", index=False)
print("✅ 11 月預測完成，已產出：2019_Nov_full.csv")

# 5. 預測 10 月數據 (作為模型驗證)
X_oct = train_df[features]
train_df['Purchase_Probability'] = rf.predict_proba(X_oct)[:, 1]
train_df.to_csv(r"C:\Users\user\Desktop\Churn_Prediction_List_Oct.csv", index=False)
print("✅ 10 月驗證清單已產出：2019_oct_full.csv")

# 6. [進階] 查看特徵重要性 (簡報分析必備)
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- 特徵重要性排行 ---")
print(importances)

# 1. 看精準度與召回率
y_pred = rf.predict(X_train) # 用訓練集自己考自己
print(classification_report(y_train, y_pred))


