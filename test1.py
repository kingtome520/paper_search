from langchain_community.llms import Ollama
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import arxiv

# 1. 初始化本地 Qwen-3 模型
llm = Ollama(
    model="qwen:3",  # Ollama 中的模型名称
    base_url="http://localhost:11434",
    temperature=0.3,
    num_predict=512
)


# 2. 创建 arXiv 检索器
class ArxivRetriever:
    def __init__(self, top_k=3):
        self.top_k = top_k

    def search(self, query: str):
        """检索 arXiv 论文并返回文档"""
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
                "pdf_url": result.pdf_url
            }
            results.append(doc)
        return results


# 3. 文档处理流水线
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


# 4. 提示工程模板
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

# 5. 构建 RAG 流水线
retriever = ArxivRetriever(top_k=3)

rag_chain = (
        {
            "context": RunnableLambda(lambda x: process_docs(retriever.search(x["question"]))),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
)

# 6. 使用示例
question = "解释transformer架构在大型语言模型中的作用及其最新进展"
response = rag_chain.invoke({"question": question})
print(response)
