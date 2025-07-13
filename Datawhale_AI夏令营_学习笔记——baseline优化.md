
# 📘 Datawhale AI夏令营：短视频评论商品识别与情感分析实战

> 本笔记总结了基于短视频评论数据的商品识别、情感分类与评论聚类分析流程，依托 `pandas`、`scikit-learn` 和 `jieba` 等工具完成建模与评估，并支持一键生成提交结果与压缩包。适用于 Datawhale AI 夏令营中「带货评论用户洞察分析挑战赛」的任务要求。

---

## 📌 1. 项目目标与任务拆解

本项目围绕三个核心任务展开：

1. **商品识别**：从视频描述中识别所推广的商品。
2. **情感分析**：对评论文本进行多维度分类，包括情感倾向、是否场景相关、是否提问或建议。
3. **评论聚类**：按不同维度对评论进行主题聚类，提炼用户关注焦点。

最终生成 `submit_videos.csv` 和 `submit_comments.csv`，并打包为压缩文件。

---

## 🧰 2. 核心依赖与数据加载

```python
import pandas as pd
import numpy as np
import jieba
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
```

加载清洗后的视频与评论数据：

```python
video_data = pd.read_csv("origin_videos_data.csv")
comments_data = pd.read_csv("origin_comments_data.csv")
```

---

## ✨ 3. 商品识别模块

**目标**：基于视频描述预测 `product_name`。

### ➤ 文本清洗与训练集划分：

```python
def clean_text(text):
    return re.sub(r"[^\w\s]", "", str(text)).lower()

video_data["clean_text"] = video_data["video_desc"].apply(clean_text)
train_df = video_data[~video_data["product_name"].isnull()]
X_train, X_val, y_train, y_val = train_test_split(train_df["clean_text"], train_df["product_name"], test_size=0.2)
```

### ➤ TF-IDF + LogisticRegression 训练分类器：

```python
product_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=jieba.lcut, max_features=300, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])
product_pipeline.fit(X_train, y_train)
```

---

## ❤️ 4. 多维情感分析模块

**目标**：预测以下四个标签：

- `sentiment_category`：正面 / 负面 / 中性等
- `user_scenario`：是否与使用场景相关
- `user_question`：是否为提问
- `user_suggestion`：是否为建议

### ➤ 统一建模流程：

```python
for col in ['sentiment_category', 'user_scenario', 'user_question', 'user_suggestion']:
    clf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=jieba.lcut, ngram_range=(1,2), max_features=300)),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    labeled = comments_data[~comments_data[col].isnull()]
    clf_pipeline.fit(labeled["comment_text"], labeled[col])
    comments_data[col] = clf_pipeline.predict(comments_data["comment_text"])
```

---

## 🔄 5. 评论聚类模块

**目标**：根据情感/场景/提问/建议的类别筛选评论，进行聚类并输出主题关键词。

### ➤ 自动选择最佳聚类数（轮廓系数）：

```python
def optimal_kmeans_clustering(texts, k_range=(5,8)):
    ...
    for k in range(k_range[0], k_range[1]+1):
        ...
        score = silhouette_score(tfidf_matrix, cluster_labels)
    ...
    return [top_terms[label] for label in best_labels]
```

### ➤ 多维聚类执行：

```python
cluster_map = {
    "positive_cluster_theme": comments_data["sentiment_category"].isin([1,3]),
    ...
}

for col, condition in cluster_map.items():
    comments_data.loc[condition, col] = optimal_kmeans_clustering(
        comments_data.loc[condition, "comment_text"]
    )
```

---

## 📤 6. 文件导出与打包提交

生成提交文件：

```python
video_data[["video_id", "product_name"]].to_csv("submit_videos.csv", index=False)
comments_data.to_csv("submit_comments.csv", index=False)
```

压缩打包：

```python
import zipfile
with zipfile.ZipFile("optimized_submit.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("submit_videos.csv", arcname="submit/submit_videos.csv")
    zipf.write("submit_comments.csv", arcname="submit/submit_comments.csv")
```

---

## 🧠 7. 小结与提升建议

| 模块 | 优化方向 |
|------|----------|
| 商品识别 | 支持多模型融合，如 SVM / XGBoost |
| 情感分析 | 考虑多任务学习，使用 RoBERTa |
| 聚类聚焦 | 可添加 UMAP 降维可视化聚类效果 |
| 数据增强 | 使用回译、同义词替换增强训练样本 |
