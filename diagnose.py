# diagnose.py
import logging
from RAG import initialize_rag_system
from RAG.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rag():
    # 1. 初始化RAG
    logger.info("=== 初始化RAG系统 ===")
    retriever = initialize_rag_system()
    
    # 2. 测试检索
    logger.info("\n=== 测试检索 ===")
    queries = [
        "什么是CodeAID",
        "a标签是干嘛用的",
        "What is CodeAID"
    ]
    
    for query in queries:
        logger.info(f"\n查询: '{query}'")
        context = retriever.retrieve(query)
        print(f"结果: {context[:1000]}...")  
if __name__ == "__main__":
    test_rag()