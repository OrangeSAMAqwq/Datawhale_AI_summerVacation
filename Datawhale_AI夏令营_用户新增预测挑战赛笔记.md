# Datawhale AI夏令营 - 用户新增预测挑战赛笔记

## 📌 任务简介

本赛题为 **二分类问题**，目标是预测某用户是否为新增用户（字段 `is_new_did`）。使用的是科大讯飞提供的用户行为数据，包括设备、网络、时间戳、行为模块、应用渠道等多字段组成的结构化数据。

- **评价指标**：F1 Score（用于处理正负样本不均衡）
- **建模工具**：LightGBM + K折交叉验证
- **数据类型**：行为日志 + JSON嵌套字段

---

## 🔧 数据预处理与特征工程

### 1. 合并训练与测试数据
统一进行清洗和编码：

```python
full_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
```

### 2. 时间特征提取
将时间戳 `common_ts` 转换为：
- `hour` 小时
- `day` 日
- `dayofweek` 周几

### 3. JSON字段解析
`udmap` 字段为 JSON 格式，提取 `botId` 与 `pluginId` 两个字段后进行编码。

### 4. 类别特征编码
- 对高基数类别进行 Top-N 筛选
- 低频类别统一编码为 `"other"`，防止长尾影响模型

### 5. 用户行为聚合特征（以 `did` 为单位）
聚合统计如下指标：

- `mid/eid` 的唯一计数和总计数
- 用户活跃小时均值、标准差
- 活跃天数等

```python
df.groupby('did').agg({
    'mid': ['nunique', 'count'],
    ...
})
```

---

## ⚙️ 模型构建与训练

### LightGBM 模型参数：

```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 63,
    'max_depth': 10,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    ...
}
```

### Stratified K-Fold（5折交叉验证）
使用 `StratifiedKFold` 保证每折中标签分布一致。

每一折都会：
- 拟合一个模型
- 用于对验证集预测 + 阈值优化
- 同时预测测试集并加权平均（5次）

---

## 📐 阈值优化

默认二分类阈值为 `0.5`，但在样本不均衡情况下并非最优。

通过如下函数对多个候选阈值（0.05~0.95）搜索最优 F1 分数：

```python
def find_optimal_threshold(y_true, y_pred_proba):
    ...
    return best_th, best_f1
```

---

## 📊 模型评估与提交

- 使用每一折最优阈值的平均作为整体预测阈值
- 输出 OOF（Out-of-Fold）F1 分数
- 生成提交文件 `submit_optimized.csv`

```python
df_submit['is_new_did'] = (test_preds >= avg_th).astype(int)
```

---

## ✅ 总结与亮点

| 模块 | 优化点 |
|------|--------|
| 特征工程 | JSON解析、多粒度时间提取、用户行为聚合 |
| 编码 | 高基数类别Top-N筛选 + LabelEncoding |
| 建模 | 使用LightGBM + 5折交叉验证，泛化性好 |
| 评估 | 使用F1阈值动态优化，适应不平衡数据 |
| 输出 | 支持概率输出与提交结果二合一 |

---

## 📎 拓展建议

- 引入目标编码、频次编码进一步挖掘类别信息
- 引入伪标签提升训练集规模
- 多模型融合（如 CatBoost / TabNet）
- 用 Optuna 自动调参
