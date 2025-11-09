import gradio as gr
from huggingface_hub import InferenceClient
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import re
import nltk
from typing import List, Tuple,Optional, Dict, Any
from dataclasses import dataclass

import bs4
import getpass
import os
from dotenv import load_dotenv

import numpy as np
from collections import deque
from typing import Any
load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "true"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")






# =========================
# ✅ 定义 LangGraph 的记忆系统
# =========================
# InMemorySaver 用于在运行时保存 agent 的上下文（比如对话历史）
# 如果你需要在不同会话间持久化，可以换成 RedisSaver 或 SQLiteSaver

checkpointer = InMemorySaver()

# =========================
# ✅ 初始化大语言模型
# =========================
# 使用 gpt-4o-mini 模型（比 gpt-4o 更快更便宜）

model=init_chat_model(
         "gpt-4o-mini",
          temperature=0.5,
          timeout=10,
          max_tokens=1000
    )

# =========================
# ✅ 定义输出数据结构（Response Schema）
# =========================
# 这部分定义了 Agent 输出的结构化格式，用于类型化返回结果

@dataclass(slots=True)
class ResponseFormat:
    """
    Agent 结构化响应载荷定义（Stable Contract Layer）。

    字段定义：
        content (str):
            - 主内容输出（Required）
            - 语义完整、确定、可直接呈现给最终用户或上层链路
            - 必须是 deterministic text，不得返回 prompt hint 形式

        reasoning_trace (Optional[str]):
            - 可选内部分析摘要（低粒度 Summary，不是 chain-of-thought）
            - 用于调试 / 观察模型决策趋势
            - 若无意义 → None
        tools_used (Optional[List[str]]):
            - 调用的工具名称列表
            
    """
    content: str
    reasoning_trace: Optional[str] = None
    tools_used: Optional[List[str]] = None  # 调用的工具





# =========================
# ✅ 定义rag向量存储
# =========================


vector_store = InMemoryVectorStore(embeddings)






bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"爬出来了！Total characters: {len(docs[0].page_content)}")




class SemanticCache:
    """
    语义缓存（简单 LRU + embedding 相似度判断）。
    存储结构：
      - queries: 原始 query 文本（list）
      - embeddings: numpy array list
      - responses: 序列化字符串（或你希望缓存的任意对象）
      - docs: 对应的检索到的 Document 列表（可选）
      - order: deque, 用于实现 LRU 淘汰（保存索引）
    使用方法：
      cache = SemanticCache(embeddings_model=embeddings, max_size=128, threshold=0.88)
      hit = cache.get(query)  # 如果命中返回 dict 否则 None
      cache.add(query, serialized, retrieved_docs)
    """
    def __init__(self, embeddings_model, max_size: int = 128, threshold: float = 0.88):
        self.emb_model = embeddings_model
        self.max_size = max_size
        self.threshold = float(threshold)  # 相似度阈值（cosine）
        self.queries: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.responses: list[Any] = []    # 存放序列化结果（string 或其他）
        self.docs: list[Any] = []         # 存放对应的 retrieved docs（Document 列表）
        self.order = deque()              # 保存索引，实现 LRU：右侧为最近使用

    def _embed(self, text: str) -> np.ndarray:
        # OpenAIEmbeddings 通常有 embed_documents 或 embed_query
        # 我们优先尝试 embed_documents（返回 list），否则尝试 embed_query
        try:
            emb = self.emb_model.embed_documents([text])[0]
        except Exception:
            emb = self.emb_model.embed_query(text)  # 某些实现存在该方法
        return np.array(emb, dtype=float)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def find_best(self, emb: np.ndarray):
        """返回 (best_idx, best_score) 或 (None, 0.0)"""
        if len(self.embeddings) == 0:
            return None, 0.0
        best_idx = None
        best_score = -1.0
        # 线性扫描：缓存条目通常不多（max_size <= 1024），足够快
        for i, e in enumerate(self.embeddings):
            score = self._cosine(emb, e)
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx, best_score

    def get(self, query: str):
        """尝试命中缓存。命中返回 dict: {'query', 'response', 'docs', 'score'}"""
        emb = self._embed(query)
        best_idx, best_score = self.find_best(emb)
        if best_idx is not None and best_score >= self.threshold:
            # 更新 LRU：把 best_idx 移到右端（最近使用）
            try:
                self.order.remove(best_idx)
            except ValueError:
                pass
            self.order.append(best_idx)
            return {
                "query": self.queries[best_idx],
                "response": self.responses[best_idx],
                "docs": self.docs[best_idx],
                "score": best_score,
            }
        return None

    def add(self, query: str, response: Any, docs: Any):
        """把新条目加入缓存，若超过 max_size 则淘汰最旧的条目"""
        emb = self._embed(query)
        # 添加
        idx = len(self.queries)
        self.queries.append(query)
        self.embeddings.append(emb)
        self.responses.append(response)
        self.docs.append(docs)
        self.order.append(idx)
        # 淘汰超出上限的最旧项
        while len(self.order) > self.max_size:
            old_idx = self.order.popleft()
            # 标记删除：将位置留空以保持索引稳定（简单实现）
            # 这里我们将对应条目置为 None; 下次 find_best 时会跳过 None
            self.queries[old_idx] = None
            self.embeddings[old_idx] = np.zeros_like(emb) * 0.0
            self.responses[old_idx] = None
            self.docs[old_idx] = None

    def stats(self):
        total = sum(1 for q in self.queries if q is not None)
        return {"capacity": self.max_size, "entries": total, "threshold": self.threshold}
    
    
# 创建语义缓存实例
semantic_cache = SemanticCache(embeddings_model=embeddings, max_size=128, threshold=0.3) # 创建语义缓存实例



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"分割博客文章为 {len(all_splits)} 个子文档。")

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

def simple_tokenize(text: str):
    # 去掉多余换行，分割标点与空白
    text = re.sub(r'\s+', ' ', text.strip())
    # 简单按空白分词，针对英文/代码/混合文本效果合理
    return text.split()

# 如果你安装了 nltk 并想要更强分词（英文），可以改用：
# from nltk.tokenize import word_tokenize
# def nltk_tokenize(text: str):
#     return word_tokenize(text)

# 构建语料：把 all_splits 的 page_content 当作 document text
corpus_texts = [d.page_content for d in all_splits]

# 生成 tokens 列表
corpus_tokens = [simple_tokenize(t) for t in corpus_texts]

# 创建 BM25 索引
bm25 = BM25Okapi(corpus_tokens)

# 可选：保存反向索引或元数据映射（这里我们用索引号直接对应 all_splits）
# doc_index -> all_splits[doc_index]
print(f"BM25 索引构建完成，文档数: {len(corpus_tokens)}")


def bm25_retrieve(query: str, top_k: int = 5, tokenizer=simple_tokenize) -> List[Tuple[int, float]]:
    """
    返回 [(doc_index, score), ...]，按 score 降序
    """
    q_tokens = tokenizer(query)
    scores = bm25.get_scores(q_tokens)  # numpy array
    # 取 top_k 索引（score 可能为 0 或负）
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]



# =========================
# ✅ 定义可供 Agent 使用的工具函数（Tools）
# =========================
# @tool(response_format="content_and_artifact")
# def retrieve_context(query: str):
#     """Retrieve information to help answer a query."""
#     retrieved_docs = vector_store.similarity_search(query, k=2)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata}\nContent: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs     #旧版本的

# =========================


# bm25_k	BM25 检索的候选数量	10
# dense_k	向量检索的候选数量	5
# bm25_threshold	触发 Dense 的门槛（最高分低于此值时触发）	2.0～3.0

@tool(response_format="content_and_artifact")
def retrieve_context(query: str, bm25_k: int = 10, dense_k: int = 5, bm25_threshold: float = 2.0):
    """
    Cascade Retrieval:
      1. 优先命中语义缓存
      2. BM25 检索 → 若得分 >= 阈值则直接返回
      3. 若得分低于阈值 → 再调用 Dense 向量检索
      4. 所有结果都会写入语义缓存
    """

    # 1️⃣ 语义缓存命中
    hit = semantic_cache.get(query)
    if hit is not None:
        print(f"[Cache] HIT (score={hit['score']:.3f}) for query: {query[:50]}...")
        return hit["response"], hit["docs"]

    print("[Cache] MISS -> Starting cascade retrieval...")

    # 2️⃣ BM25 检索阶段
    bm25_results = bm25_retrieve(query, top_k=bm25_k)
    bm25_scores = [s for _, s in bm25_results]
    avg_score = np.mean(bm25_scores) if bm25_scores else 0.0
    max_score = np.max(bm25_scores) if bm25_scores else 0.0
    print(f"[BM25] avg_score={avg_score:.3f}, max_score={max_score:.3f}")

    bm25_indices = [i for i, _ in bm25_results]
    bm25_docs = [all_splits[i] for i in bm25_indices]

    # 3️⃣ 判断是否触发 Dense 检索
    if max_score >= bm25_threshold:
        print(f"[Cascade] BM25 score sufficient (>= {bm25_threshold}). Skip Dense retrieval.")
        merged_docs = bm25_docs
    else:
        print(f"[Cascade] BM25 score too low (< {bm25_threshold}). Triggering Dense retrieval...")
        dense_docs = vector_store.similarity_search(query, k=dense_k)

        # 合并（去重）
        seen = set()
        merged_docs = []
        for d in bm25_docs + dense_docs:
            key = getattr(d, "page_content", None)
            if key not in seen:
                seen.add(key)
                merged_docs.append(d)

    # 4️⃣ 序列化 + 缓存写入
    serialized = "\n\n".join(
        (f"Source: {d.metadata}\nContent: {d.page_content}") for d in merged_docs
    )

    semantic_cache.add(query, serialized, merged_docs)
    print(f"写入了缓存！当前缓存状态: {semantic_cache.stats()}")

    return serialized, merged_docs






# =========================
# ✅ 定义自定义上下文（Context Schema）
# =========================
# Agent 执行时可以访问这个上下文，比如用户 ID、权限、配置等

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# =========================
# ✅ 创建智能代理（Agent）
# =========================
# create_agent 用于组装一个可执行的智能体，
# 它能自动根据系统提示、工具和上下文来规划任务。
prompt = """
你是一个智能 AI 助手，专注于回答与人工智能、机器学习、大语言模型（LLM）、RAG 和 LangChain 相关的问题。
你的目标是用简洁、准确且有帮助的中文回答用户的问题。

### 规则
- 你必须始终用中文回答用户的问题。
- 你可以调用 `retrieve_context` 工具 **最多一次** 来检索相关信息。
- 如果检索工具返回了上下文，请使用它来生成最终的简短答案。
- 不要进入无限循环、重复调用工具或重复自我查询。
- 始终以 `ResponseFormat` 中定义的格式返回答案。
- 如果用户询问博客内容，请使用 `retrieve_context` 获取博客中的相关信息。

### 输出格式提醒
- 始终返回 `ResponseFormat` 结构化格式。


### 目标
高效回答用户的问题。如果不确定答案 → 检索一次 → 整合 → 回答。
如果有信心 → 直接回答。
"""


agent = create_agent(
    model=model,                        # 语言模型
    system_prompt=prompt,  # 系统角色提示
    tools=[retrieve_context],           # 可用工具
    context_schema=Context,             # 上下文类型定义
    response_format=ResponseFormat,     # 输出格式定义
    checkpointer=checkpointer           # 内存检查点（存储对话状态）
)

# =========================
# ✅ 定义 Agent 调用配置
# =========================
# “thread_id” 用于标识一个连续的会话线程（支持多轮记忆）

config = {"configurable": {"thread_id": "1"}}





def run_agent_conversation(agent, user_inputs: list[str], context: Context, config: dict):
    """
    运行多轮交互的函数版本。
    """
    responses = []
    for user_input in user_inputs:
        print(f"\n用户输入: {user_input}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            context=context
        )

        # 结构化返回：按 ResponseFormat 使用
        structured = response.get("structured_response")
        if structured:
            try:
                print("\n助手:", structured.content)
                if structured.reasoning_trace:
                    print("思维链:", structured.reasoning_trace)
                if structured.tools_used:
                    print("调用的工具:", structured.tools_used)
                responses.append({
                    "content": structured.content,
                    "reasoning_trace": structured.reasoning_trace,
                    "tools_used": structured.tools_used
                })
            except Exception:
                print("\n助手(完整结构):", structured)
                responses.append({"content": str(structured)})
        else:
            print("\n助手（原始响应）:", response)
            responses.append({"content": str(response)})
    return responses


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    hf_token: gr.OAuthToken,
):
    """
    Gradio 的响应函数，调用 run_agent_conversation 实现多轮对话。
    """
    context = Context(user_id="1")

    # 将当前输入作为用户输入
    user_inputs = [message]
    responses = run_agent_conversation(agent, user_inputs, context, config)

    # 返回最后一轮的响应
    return responses[-1] if responses else "Error: No response generated."


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
