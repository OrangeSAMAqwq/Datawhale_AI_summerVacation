
# 📘 Datawhale AI夏令营：短视频评论商品识别与情感分析实战学习笔记

> 作者：@hakimengana228  
> 时间：2025年7月  
> 项目：基于评论的用户洞察分析挑战赛  
> 标签：文本挖掘、机器学习、情感分析、聚类、数据竞赛

---

## 🏁 学习背景 & 初衷

在报名 Datawhale AI 夏令营之初，我抱着“把项目做完、把代码跑通”的目标。但深入参与后，我发现这次挑战不仅仅是编程任务，更是一次数据理解与算法设计能力的综合锻炼。

通过拆解赛事任务、阅读 baseline、优化建模流程，我逐渐建立起了完整的文本分析 pipeline，从“看得懂”到“能实现”，再到“尝试改进”，这段学习过程极大提升了我对真实业务场景中 NLP 技术的掌握力。

---

## 📌 项目目标与任务拆解

本项目任务围绕带货视频及评论数据展开，目标是通过自然语言处理手段，实现从非结构化数据中提取结构化商业洞察。

三大子任务如下：

1. 🎯 **商品识别**：从视频文本中识别推广商品（多为品牌名/产品型号）；
2. ❤️ **情感分析**：对评论打标签，包括情感倾向、是否为建议、提问或使用场景；
3. 🧠 **评论聚类**：提炼用户意见主题词，帮助品牌洞察消费偏好。

---

## 🧰 核心依赖 & 数据加载

### 🔧 依赖工具

- `pandas`：数据读取与操作
- `jieba`：中文分词
- `scikit-learn`：TF-IDF、分类器、聚类模型
- `zipfile`：压缩提交结果

```python
import pandas as pd
import jieba
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
```

### 📁 数据结构说明

| 数据集 | 描述 |
|--------|------|
| `origin_videos_data.csv` | 每条短视频的描述与商品名（部分标注） |
| `origin_comments_data.csv` | 评论文本及其标签信息（部分标注） |

---

## 🛠 商品识别模块

这一阶段我先用 TF-IDF + LogisticRegression 实现了商品名的预测。

### ✂ 文本预处理

```python
def clean_text(text):
    return re.sub(r"[^\w\s]", "", str(text)).lower()
```

清洗后的视频描述被用于训练分类器。通过交叉验证划分训练/验证集，评估分类性能。

### 🔍 模型结构

```python
product_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(tokenizer=jieba.lcut, max_features=300, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])
```

我也尝试过替换成 SGDClassifier，但稳定性不如 LogReg。

---

## 💬 多维情感分析模块

起初我以为只要分类情感正负就行，但深入理解数据后发现还需要处理用户场景、建议、疑问等维度 —— 这是更贴近真实商业反馈的设计。

### 🧩 四维标签

| 标签字段 | 含义 |
|----------|------|
| `sentiment_category` | 情感极性（正/负/中性/混合） |
| `user_scenario` | 是否涉及使用场景 |
| `user_question` | 是否为用户提问 |
| `user_suggestion` | 是否提出建议 |

### 🏗 建模逻辑一致性复用

我将四个分类任务封装为同一流程，极大减少了冗余代码，也方便日后替换为 BERT 统一多任务建模。

---

## 🔎 评论聚类模块

这是我最费时间也最有收获的部分。项目要求对每种评论类型分别聚类，提炼“类簇主题词”。

### 🤔 起初的问题

一开始我使用固定的 `n_clusters=5` 聚类所有数据，但发现主题词非常混乱。

于是我加入了**轮廓系数（Silhouette Score）**自动调参逻辑，提升了聚类的质量与一致性。

```python
def optimal_kmeans_clustering(texts, k_range=(5,8)):
    ...
    score = silhouette_score(tfidf_matrix, cluster_labels)
    ...
```

每个聚类的中心词我都逐条打印分析，用实际语言判断其语义聚合性，也让我真正理解了“主题词”不是靠模型生成，而是靠我们设计好提取机制与语料结构。

---

## 📤 提交与压缩导出

最后我通过 Pandas 将结果保存为：

- `submit_videos.csv`
- `submit_comments.csv`

再利用 zipfile 模块打包：

```python
with zipfile.ZipFile("optimized_submit.zip", 'w') as zipf:
    zipf.write("submit_videos.csv", arcname="submit/submit_videos.csv")
    zipf.write("submit_comments.csv", arcname="submit/submit_comments.csv")
```

---

## 🧠 总结与反思

| 阶段 | 收获 |
|------|------|
| 任务理解 | 从业务视角出发构建 NLP 流程 |
| 特征工程 | TF-IDF + n-gram 在中文任务中的实践 |
| 模型调参 | 尝试 LogisticRegression、SVM、SGD 等替换方案 |
| 聚类分析 | 掌握 silhouette score、聚类数调优、主题词提取方法 |
| 工程集成 | 学会一站式 pipeline 构建与结果导出封装 |

最重要的是，这不仅是一次模型构建训练，更是一次**面向应用场景的数据分析实战**。

---

## 🚀 后续计划

- 将分类模型替换为中文 BERT，提升语义理解能力；
- 尝试加入 `TextRank`、`LDA` 做主题建模增强聚类解释性；
- 多任务学习结构，将四个分类器合并为一个共享语义网络。

---

感谢 Datawhale 的组织与支持，这是我第一次真正沉浸式地做一个 NLP 项目。它不仅提升了我对技术的掌握，也让我更加相信数据驱动洞察的价值。

