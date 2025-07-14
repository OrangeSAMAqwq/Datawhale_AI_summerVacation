# Datawhale AI夏令营 - 用户新增预测挑战赛进阶笔记

## 🚀 项目背景

本项目源于讯飞开放平台的真实业务挑战，目标是基于用户行为数据预测某用户是否为新增用户（`is_new_did`），属于经典的 **不平衡二分类问题**。优化后模型已从 F1 ≈ 0.86 提升至 F1 ≈ 0.93。

---

## 🔧 数据预处理与高级特征工程

### 🕒 时间特征增强
- 将 `common_ts` 转换为：
  - `hour`, `day`, `dayofweek`
  - `time_bucket`：将一天分为 4 个时间段用于行为分段

### 🔍 JSON字段结构化
- `udmap` 字段为 JSON 格式，提取 `botId`, `pluginId` 作为单独特征使用

### 🔁 组合交叉特征
- `device_os = device_brand + os_type`：反映设备与系统的联动偏好

### 🧮 用户聚合特征（基于 `did`）
- 聚合字段包括：`eid`, `mid`, `hour`, `common_ts` 等
- 派生特征如：
  - `base_events_per_mid`：平均事件频率
  - `base_activity_variation`：行为多样性

---

## ⚙️ 模型构建与训练逻辑

### 🧠 使用模型
- LightGBM，天然支持类别变量、高效处理结构化数据

### 🎯 模型参数（优化后）

```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 10,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    ...
}
```

### 🧪 交叉验证方案
- `StratifiedKFold(n_splits=5)`
- 每折预测 `X_test` 后进行加权平均（模型集成）

---

## 🔍 阈值优化与评估策略

### 📐 动态阈值搜索
```python
np.arange(0.05, 0.95, 0.01)
```
- 在验证集上找到最佳 F1 对应的概率阈值

### 📊 模型评估指标
- 平均 F1（5折）
- OOF（Out-of-Fold）预测 F1

---

## 📝 提交文件构建

```python
df_submit['is_new_did'] = (test_preds >= avg_th).astype(int)
df_submit['proba'] = test_preds
df_submit.to_csv('submit_advanced.csv')
```

字段说明：
- `is_new_did`：最终预测标签
- `proba`：预测概率值（可用于 ensemble）

---

## ✅ 最终结果

| 指标 | 数值 |
|------|------|
| CV 平均 F1 | ~0.93 |
| 提交文件名 | `submit_advanced.csv` |
| 模型数量 | 5（交叉验证集成） |

---

## 💡 可进一步优化方向

- 引入目标编码（Target Encoding）
- 使用 TF-IDF 表达类别关键词差异
- 进行两阶段建模（用户层+事件层）
- 半监督学习策略：利用伪标签强化模型训练
- 尝试模型融合（如 Stacking + CatBoost）

---

📚 项目代码文件：`LightGBM 优化增强版 baseline`  
🎓 建议结合 Notebook 或 wandb 做可视化追踪

