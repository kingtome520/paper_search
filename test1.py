from flask import Flask, request, jsonify
from flask_cors import CORS
import arxiv
import json
import os
import traceback
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='web', static_url_path='/')
CORS(app)

from langchain_ollama import OllamaLLM

# 使用更快的模型
llm = OllamaLLM(
    model="qwen2.5:0.5b",  # 更小的模型，推理更快
    base_url="http://localhost:11434",
    temperature=0.7,
    num_thread=8,  # 使用更多线程
)


class ArxivRetriever:
    def __init__(self, top_k=2):  # 减少返回结果数量
        self.top_k = top_k

    def search(self, query: str):
        """检索 arXiv 论文"""
        logger.info(f"Searching arxiv for: {query}")
        start_time = time.time()

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

            elapsed = time.time() - start_time
            logger.info(f"arXiv search completed in {elapsed:.2f}s, found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching arxiv: {str(e)}")
            return []


# 缓存检索结果
@lru_cache(maxsize=50)
def cached_arxiv_search(query: str):
    retriever = ArxivRetriever(top_k=2)
    return retriever.search(query)


@app.route('/api/search', methods=['POST'])
def search_papers():
    start_time = time.time()

    try:
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'error': '问题不能为空'}), 400

        logger.info(f"Processing search: {question}")

        # 并行处理：同时进行arXiv搜索和AI分析
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交并行任务
            search_future = executor.submit(cached_arxiv_search, question)

            # 简化的AI响应（可选，如果模型仍然很慢）
            ai_response = f"关于 '{question}' 的搜索结果如下。以下是相关的学术论文："

            # 获取搜索结果
            papers = search_future.result(timeout=15)  # 15秒超时

            # 如果有论文，添加到AI响应中
            if papers:
                for i, paper in enumerate(papers, 1):
                    ai_response += f"\n\n[{i}] {paper['title']}\n作者: {', '.join(paper['authors'][:3])}\n年份: {paper['year']}"

        elapsed = time.time() - start_time
        logger.info(f"Request completed in {elapsed:.2f}s")

        return jsonify({
            'response': ai_response,
            'papers': papers
        })

    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)