from flask import Flask, request, jsonify
from flask_cors import CORS
import arxiv
import json
import os
import traceback
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor


# 设置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='web', static_url_path='/')
CORS(app)  # 允许跨域请求

from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 初始化 Ollama 连接
llm = OllamaLLM(
    model="qwen3:8b",  # Ollama 中的模型名称
    base_url="http://localhost:11434",  # Ollama 服务地址
    temperature=0.7,  # 控制随机性 (0~1)
)


# 创建 arXiv 检索器
class ArxivRetriever:
    def __init__(self, top_k=3):
        self.top_k = top_k

    def search(self, query: str):
        """检索 arXiv 论文并返回文档"""
        logger.debug(f"Searching arxiv for query: {query}")
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=self.top_k,
                sort_by=arxiv.SortCriterion.Relevance
            )

            results = []
            for result in client.results(search):
                doc = {
                    "title": result.title,
                    "authors": [a.name for a in result.authors],
                    "summary": result.summary,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "entry_id": result.entry_id,
                    "pdf_url": result.pdf_url,
                    "year": result.published.year
                }
                results.append(doc)

            logger.debug(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching arxiv: {str(e)}")
            logger.error(traceback.format_exc())
            raise


# 文档处理流水线
def process_docs(retrieved_docs):
    """处理检索到的文档并生成带引用的上下文"""
    processed = []
    for i, doc in enumerate(retrieved_docs):
        # 创建带引用的文本块
        content = f"""
        [引用 {i + 1}]
        标题: {doc['title']}
        作者: {', '.join(doc['authors'])}
        发布日期: {doc['published']}
        PDF链接: {doc['pdf_url']}

        摘要:
        {doc['summary']}
        """
        processed.append(content)
    return "\n\n".join(processed)


# 优化查询的提示词模板
query_optimization_template = """
你是一个专业的学术搜索助手。你的任务是优化用户的查询，使其更适合在arXiv学术数据库中搜索。
请将用户的查询转换为更精确、更适合学术搜索的关键词组合。

用户查询: {question}

请按照以下格式回复:
优化后的查询: [你的优化查询]

优化原则:
1. 使用英文关键词（如果原查询是中文，请翻译成英文）
2. 添加相关的学术术语
3. 保持简洁但具体
4. 避免过于宽泛的词汇
"""

query_optimization_prompt = ChatPromptTemplate.from_template(query_optimization_template)

# 提示工程模板
template = """
你是一个专业科学助手，请基于提供的上下文信息回答问题。
回答必须包含引用标记（如 [1], [2]）并附上参考文献列表。

上下文:
{context}

问题: {question}

请按以下格式回答:
[答案正文]
[引用标记] 对应上下文中的引用编号

参考文献:
[1] 标题 - 作者 (年份) [链接]
[2] ...
"""
prompt = ChatPromptTemplate.from_template(template)

# 构建查询优化链
query_optimizer = query_optimization_prompt | llm | StrOutputParser()

# 构建 RAG 流水线
retriever = ArxivRetriever(top_k=3)

def optimize_and_search(inputs):
    """优化查询并搜索"""
    question = inputs["question"]
    
    # 先优化查询
    try:
        optimized_query = query_optimizer.invoke({"question": question})
        # 提取优化后的查询
        if "优化后的查询:" in optimized_query:
            optimized_query = optimized_query.split("优化后的查询:")[1].strip()
        logger.debug(f"Original query: {question}")
        logger.debug(f"Optimized query: {optimized_query}")
    except Exception as e:
        logger.error(f"Error optimizing query: {str(e)}")
        # 如果优化失败，使用原始查询
        optimized_query = question
    
    # 使用优化后的查询进行搜索
    papers = retriever.search(optimized_query)
    context = process_docs(papers)
    return {"context": context, "question": question}

rag_chain = (
    RunnableLambda(optimize_and_search)
    | prompt
    | llm
    | StrOutputParser()
)


# 提供前端页面
@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/api/search', methods=['POST'])
def search_papers():
    try:
        data = request.get_json()
        question = data.get('question', '')

        if not question:
            return jsonify({'error': '问题不能为空'}), 400

        logger.debug(f"Received search request with question: {question}")

        # 使用RAG系统获取答案
        response = rag_chain.invoke({"question": question})

        # 同时获取论文信息用于前端显示（使用原始查询）
        papers = retriever.search(question)

        return jsonify({
            'response': response,
            'papers': papers
        })

    except Exception as e:
        logger.error(f"Error in search_papers: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)