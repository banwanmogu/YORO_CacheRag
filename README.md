# 🧠YORO ——you only rag once—— LRU缓存 —— 面对现实场景重复问题，仅需rag一次！——动态缓存替换算法

[🇨🇳 中文](README.md) | [🇺🇸 English](README_ENG.md)

# webui 1.0版现已发布在webui分支！速速本地部署或者在线尝试：https://huggingface.co/spaces/MOGU111/YORO/tree/main

<div align="center">
  <img src="image-2.png" width="300" alt="YORO Logo" />
   </div>





# 为什么使用 YORO？

## 1️⃣ 缓存
在企业级真实场景中（如客服机器人、文档解释助手），用户需要的信息往往集中在同一个段落中。

### 普通 RAG 示例
- **Q1**：YORO 作者 mogu 今年多少岁？  
  → 模型调用 RAG，查询到：mogu 今年 23 岁。
- **Q2**：YORO 作者 mogu 目前居住在哪？  
  → 模型再次调用 RAG，查询到：mogu 目前居住在日本。
- **Q3**：YORO 作者 mogu 是程序员吗？  
  → 模型再次调用 RAG，查询到：mogu 是一个程序员。

⚠️ **问题**：非常低效！可以看到模型调用了多次rag，每次都要对向量数据库进行查询。但是信息都在rag文档中的一句话中： 


> 原文档内容：
> ```
> YORO 作者 mogu 是一个目前居住在日本的 23 岁程序员
> ```

### 使用 YORO 示例
- **Q1**：YORO 作者 mogu 今年多少岁？  
  → 模型调用 RAG 查询到：mogu 今年 23 岁  
  → 同时将原文档向量化并存入缓存：


- **Q2**：YORO 作者 mogu 目前居住在哪？  
→ 模型查询缓存 **命中**，直接返回：mogu 目前在日本
- **Q3**：YORO 作者 mogu 是程序员吗？  
→ 模型查询缓存 **命中**，直接返回：mogu 是一个程序员

✅ 高效应对真实 RAG 场景！减少重复查询，提高响应速度！

---

## 2️⃣ LRU（最近最少使用）替换算法
YORO 内置了 LRU 缓存替换算法，灵感来源于计算机内存管理：

- 当缓存已满，需要替换时，优先淘汰最近没有使用的缓存块!
- 每个缓存块都有一个类似计算机内存的 **存在位**，在调出时设置为 0，类似计算机内存的“无需重写”机制

⚡ **效果**：
- 保持缓存的高命中率
- 与实际使用场景高度契合
- 避免无效缓存占用资源


<div align="center">
  <img src="image-1.png" width="300" alt="YORO Logo" />
   </div>

> 🚀 YORO是基于**LangChain** 的项目，集成了**动态特征感知的 LRU 缓存**、和检索增强生成（RAG）—— 优化**高效语义复用**。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-✅-green?logo=chainlink)
![OpenAI](https://img.shields.io/badge/OpenAI-Embeddings-purple?logo=openai)
![License](https://img.shields.io/badge/license-MIT-orange)
![Status](https://img.shields.io/badge/status-Active-brightgreen)



## 🧩 系统架构

本项目实现了一个带有语义缓存（LRU + embedding 相似度）和检索增强生成（RAG+BM25+DENSE）的流水线。下面描述了系统的核心组件与pipeline：

### 概览

- 数据采集与分割：使用 `WebBaseLoader` 抓取网页内容，并用 `RecursiveCharacterTextSplitter` 将长文档切成若干语义块（chunk），每个 chunk 为检索最小单位。此处为演示例，可以自定义url或者自行更改使用本地文档
- 向量化与索引：对 chunk 进行嵌入（`OpenAIEmbeddings`），并存入 `InMemoryVectorStore`（
- 关键词检索（BM25）：使用 `rank_bm25.BM25Okapi` 对 corpus 做倒排检索，以提供高 recall 的候选集合。
- 稠密检索（Dense）：使用向量相似度（`vector_store.similarity_search`）做语义检索，补充或对比 BM25 的召回结果。
- Dense 重排（可选）：在 BM25/union 候选上使用 bi-encoder / cross-encoder 对候选做 rerank，提高语义匹配质量。
- 语义缓存（SemanticCache，LRU）：对最终返回的序列化上下文进行缓存，采用 embedding 相似度判定命中，用于减少重复 RAG 调用。
- 智能代理（Agent）：基于 LangGraph/LangChain 的 agent 负责接收用户输入、按规则调用 `retrieve_context`（最多一次）并以结构化格式输出结果。

### 数据流（step-by-step）

1. 用户发起查询（query）。
2. Agent 调用 `retrieve_context(query)`：
   - 先检查 `SemanticCache.get(query)`（embedding 相似度 + LRU），命中直接返回缓存结果。
   - 缓存未命中时同时触发：
     - BM25 召回 top-N（例如 50-100）候选（高召回）。
     - 向量检索（Dense）召回 top-M（例如 5-20）候选（语义召回）。
   - 合并 BM25 与 Dense 候选并去重，得到候选集合（candidate pool）。
   - 对 candidate pool 进行 Dense 重排：
   - 取重排后 top-K（5）结果序列化为上下文，写入 `SemanticCache`，返回给 Agent。
3. Agent 将检索到的上下文与系统 prompt 一起传给 LLM（ `gpt-4o-mini`），并按 `ResponseFormat` 返回结构化答案。

### 关键组件与对应文件/类

- Web 加载与分割：`WebBaseLoader` + `RecursiveCharacterTextSplitter`（在 `test.py` 中示例使用）。
- Embeddings / 向量存储：`OpenAIEmbeddings` + `InMemoryVectorStore`。
- BM25 检索：`BM25Okapi`（基于 `simple_tokenize` 的分词结果）。
- Dense reranker（建议）：使用 `sentence-transformers` 的 `CrossEncoder` 或 `SentenceTransformer` 作为 bi-encoder。
- 语义缓存：`SemanticCache`（LRU + embedding cosine 判定）实现位于 `test.py`，负责缓存 query -> (serialized_context, docs) 对。
- Agent 逻辑：LangGraph的 `create_agent` 与自定义 Tools（`retrieve_context`）。

### 调优建议

- 候选规模控制：跨编码器（cross-encoder）适合小候选池（<=100），bi-encoder 可扩展到更大池并结合 FAISS。 
- 分数融合：BM25 score 与 dense score 处于不同尺度，推荐做归一化后按权重线性融合以提升排序稳定性。
- 缓存阈值：`SemanticCache.threshold` 可调（示例代码中默认 0.3），生产环境中可通过 A/B 测试或离线评估调优。


---

## 🧱 项目结构

```plaintext
.
├── 🚀 main.py                # 入口点（智能体循环）
├── 📝 .env                   # 环境变量（OPENAI_API_KEY）
├── 📦 requirements.txt       # 依赖项
├── 📖 README.md             # 文档说明
└── 🧠 test.py  # 自定义 LRU + 嵌入缓存
```

## ⚙️ 安装步骤

```bash
# 1️⃣ 克隆仓库
git clone https://github.com/yourusername/langgraph-rag-agent.git
cd langgraph-rag-agent

# 2️⃣ 创建环境
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 3️⃣ 安装依赖
pip install -r requirements.txt

# 4️⃣ 设置环境变量
# 创建 .env 文件并添加您的 OpenAI API 密钥 如需webui版，添加huggingface秘钥
echo "OPENAI_API_KEY=sk-xxxxxx" > .env
echo "HF_TOKE=hf-xxxxxx" > .env

```

## 🧩 语义缓存示例

```python
cache.stats()
# {'capacity': 128, 'entries': 7, 'threshold': 0.3}

# 缓存命中时：
[Cache] HIT (score=0.91) for query: "LangChain memory"...
```

## 📸 预览
![alt text](webquestion.png)
![alt text](terminalresponse.png)
### 可以看到，在重复输入类似相关问题时，直接访问缓存，高效便捷



## TODO：
2.0计划：缓存数据持久化    用户逻辑导航层


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=banwanmogu/YORO_CacheRag&type=date&legend=top-left)](https://www.star-history.com/#banwanmogu/YORO_CacheRag&type=date&legend=top-left)
