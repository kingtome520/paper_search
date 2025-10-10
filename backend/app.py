from flask import Flask, request, jsonify
from flask_cors import CORS
import arxiv
import json
import os
import traceback
import logging
from openai import OpenAI

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 配置CORS以允许Netlify前端的请求
CORS(app, origins="*", supports_credentials=True)

# 初始化Ollama客户端，连接本地Ollama服务
# 确保本地已安装并运行Ollama，且已下载qwen3模型
ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
try:
    ollama_client = OpenAI(
        base_url=ollama_base_url,
        api_key='ollama'  # Ollama不需要api_key，但openai包需要提供一个值
    )
    logger.info(f"Ollama client initialized successfully with base_url: {ollama_base_url}")
except Exception as e:
    logger.error(f"Failed to initialize Ollama client: {e}")
    ollama_client = None


# 初始化一个简单的LLM（示例用），实际使用时你需要替换成一个可通过API访问的模型（如OpenAI, Together AI, DeepSeek等）
# 因为Ollama通常在本地运行，无法在Railway上使用。
class MockLLM:
    def invoke(self, text):
        # 这是一个模拟响应。在实际部署中，你应该调用一个真实的LLM API。
        if "optimize" in text:
            return "优化后的查询: machine learning"
        return f"这是一段关于'{text}'的AI生成回答。此为演示内容。在实际部署中，此处应连接真实的LLM API（如OpenAI、Claude、DeepSeek等）。"

    def __or__(self, other):
        # 简单模拟 LangChain 的链式操作
        return self


# 使用Ollama LLM（如果可用）或者模拟LLM
if ollama_client:
    class OllamaLLM:
        def __init__(self, client, model_name="qwen3:8b"):
            self.client = client
            self.model_name = model_name

        def invoke(self, text):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": text}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Error calling Ollama API: {e}")
                # 出错时回退到模拟响应
                return f"这是一段关于'{text}'的AI生成回答。此为演示内容。在实际部署中，此处应连接真实的LLM API（如OpenAI、Claude、DeepSeek等）。"

        def __or__(self, other):
            # 简单模拟 LangChain 的链式操作
            return self


    llm = OllamaLLM(ollama_client)
    logger.info("Using Ollama LLM with qwen3:8b model")
else:
    llm = MockLLM()
    logger.info("Using Mock LLM as fallback")

StrOutputParser = lambda: lambda x: x


# 创建 arXiv 检索器
class ArxivRetriever:
    def __init__(self, top_k=3):
        self.top_k = top_k

    def search(self, query: str):
        """检索 arXiv 论文并返回文档"""
        logger.info(f"Searching arxiv for query: {query}")
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

            logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching arxiv: {str(e)}")
            logger.error(traceback.format_exc())
            return []  # 出错时返回空列表


# ... [保留 process_docs, template, prompt 等函数] ...
def process_docs(retrieved_docs):
    """处理检索到的文档并生成带引用的上下文"""
    processed = []
    for i, doc in enumerate(retrieved_docs):
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
prompt = lambda x: x

# 构建 RAG 流水线
retriever = ArxivRetriever(top_k=3)


def rag_function(input_dict):
    """优化的RAG流程"""
    question = input_dict["question"]
    papers = retriever.search(question)
    context = process_docs(papers)

    # 模拟链式调用
    formatted_prompt = f"上下文:\n{context}\n\n问题: {question}\n\n请按以下格式回答:\n[答案正文]\n[引用标记] 对应上下文中的引用编号\n\n参考文献:\n[1] 标题 - 作者 (年份) [链接]\n[2] ..."
    response = llm.invoke(formatted_prompt)
    return response


@app.route('/health')
def health_check():
    """健康检查端点"""
    return jsonify({"status": "healthy", "message": "ScholarSearch API is running"}), 200


@app.route('/')
def home():
    logger.info("Home endpoint accessed")
    return jsonify({"message": "ScholarSearch API is running!", "status": "OK"})


@app.route('/api/search', methods=['POST'])
def search_papers():
    try:
        # 添加日志记录
        logger.info("Received request to /api/search")

        data = request.get_json()
        question = data.get('question', '')

        if not question:
            logger.warning("Empty question received")
            return jsonify({'error': '问题不能为空'}), 400

        logger.info(f"Received search request: {question}")

        # 使用RAG系统获取答案
        response = rag_function({"question": question})

        # 同时获取论文信息用于前端显示
        papers = retriever.search(question)

        result = {
            'response': response,
            'papers': papers
        }

        logger.info(f"Search completed successfully for: {question}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in search_papers: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)