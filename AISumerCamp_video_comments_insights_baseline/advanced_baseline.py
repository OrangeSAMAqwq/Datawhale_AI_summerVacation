# 优化增强版 LightGBM 二分类建模方案（含高级特征工程）

import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')

# 数据加载
df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./testA_data.csv')
df_submit = df_test[['did']].copy()
full_df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# 时间特征
def extract_time_features(df):
    df['ts'] = pd.to_datetime(df['common_ts'], unit='ms')
    df['hour'] = df['ts'].dt.hour
    df['day'] = df['ts'].dt.day
    df['dayofweek'] = df['ts'].dt.dayofweek
    df['time_bucket'] = pd.cut(df['hour'], bins=[0,6,12,18,24], labels=[0,1,2,3])
    df.drop('ts', axis=1, inplace=True)
    return df
full_df = extract_time_features(full_df)

# 解析 udmap 字段
def extract_udmap(df):
    def parse_json(text):
        try:
            j = json.loads(text)
            return j.get('botId', 'null'), j.get('pluginId', 'null')
        except:
            return 'null', 'null'
    df['botId'], df['pluginId'] = zip(*df['udmap'].astype(str).map(parse_json))
    return df
full_df = extract_udmap(full_df)

# 组合特征
def create_combination_features(df):
    df['device_os'] = df['device_brand'].astype(str) + '_' + df['os_type'].astype(str)
    return df
full_df = create_combination_features(full_df)

# 类别编码
def encode_categorical(df, cols, top_n=50):
    for col in cols:
        vc = df[col].value_counts()
        top_cats = vc[:top_n].index
        df[col] = df[col].apply(lambda x: x if x in top_cats else 'other')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

cat_cols = [
    'device_brand', 'ntt', 'operator', 'common_country', 'common_province', 'common_city',
    'appver', 'channel', 'os_type', 'botId', 'pluginId', 'device_os'
]
full_df = encode_categorical(full_df, cat_cols)

# 用户聚合特征
def create_user_features(df):
    agg_funcs = {
        'eid': ['count', 'nunique'],
        'mid': ['nunique'],
        'hour': ['mean', 'std'],
        'dayofweek': ['mean', 'std'],
        'day': ['nunique','mean','max','min'],
        'common_ts': ['mean','max','min',np.ptp],
        'udmap': ['nunique'],
        'device_brand': ['nunique'],
        'ntt': ['nunique'],
        'operator': ['nunique'],
        'common_city': ['nunique'],
        'appver': ['nunique'],
        'channel': ['nunique']
    }
    user_df = df.groupby('did').agg(agg_funcs)
    user_df.columns = ['base_'+'_'.join(col) for col in user_df.columns]
    user_df['base_events_per_mid'] = user_df['base_eid_count'] / (user_df['base_mid_nunique'] + 1e-5)
    user_df['base_activity_variation'] = user_df['base_eid_nunique'] / (user_df['base_eid_count'] + 1e-5)
    return user_df.reset_index()

agg_df = create_user_features(full_df)
full_df = full_df.merge(agg_df, on='did', how='left')

# LabelEncode did
le_did = LabelEncoder()
full_df['did'] = le_did.fit_transform(full_df['did'].astype(str))

# 拆分数据
train_len = len(df_train)
train_df = full_df.iloc[:train_len].copy()
test_df = full_df.iloc[train_len:].copy()
X_train = train_df.drop(['is_new_did'], axis=1).drop(columns=['udmap'])
y_train = train_df['is_new_did']
X_test = test_df.drop(columns=['is_new_did', 'udmap'])

# 最优阈值搜索
def find_optimal_threshold(y_true, y_pred_proba):
    best_th, best_f1 = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_true, (y_pred_proba >= t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_th = t
    return best_th, best_f1

# 模型训练+交叉验证
params = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'max_depth': 10, 'num_leaves': 63, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'min_child_samples': 10, 'verbose': -1, 'n_jobs': 8,
    'seed': int(time.time()) % 1000000
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
oof_preds = np.zeros(len(X_train))
oof_probas = np.zeros(len(X_train))
th_list, f1_list, models = [], [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"\n==== Fold {fold+1} ====")
    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

    lgb_train = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    lgb_val = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    )

    val_proba = model.predict(X_val)
    test_preds += model.predict(X_test) / skf.n_splits
    oof_probas[val_idx] = val_proba

    best_th, f1 = find_optimal_threshold(y_val.values, val_proba)
    oof_preds[val_idx] = (val_proba >= best_th).astype(int)

    print(f"Best Threshold = {best_th:.3f}, F1 = {f1:.5f}")
    th_list.append(best_th)
    f1_list.append(f1)
    models.append(model)

# 评估
avg_th = np.mean(th_list)
oof_f1 = f1_score(y_train, (oof_probas >= avg_th).astype(int))
print("\n===== Final Evaluation =====")
print(f"Avg Threshold: {avg_th:.4f}")
print(f"CV F1: {np.mean(f1_list):.5f}, OOF F1: {oof_f1:.5f}")

# 提交
test_labels = (test_preds >= avg_th).astype(int)
df_submit['is_new_did'] = test_labels
df_submit['proba'] = test_preds
df_submit.to_csv('submit_advanced.csv', index=False)
print("\n提交文件保存为 submit_advanced.csv")
print(f"预测新增用户比例: {test_labels.mean():.4f}")