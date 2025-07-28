# diagnose.py
import logging
from RAG import initialize_rag_system
from config import RAG_CONFIG, SERVICE_CONFIG
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rerank_results():
    """测试重排序结果"""
    logger.info("=== 测试重排序结果 ===")
    
    # 初始化RAG系统
    retriever = initialize_rag_system()
    
    # 测试查询
    test_queries = [
        "人如何自律",
        "html是什么",
        "what is codeAID",
        "荀子有哪些名言"
    ]
    
    for query in test_queries:
        logger.info(f"\n{'='*50}")
        logger.info(f"查询: '{query}'")
        logger.info(f"{'='*50}")
        
        # 测试基础检索（无重排）
        logger.info("\n--- 基础检索结果（无重排） ---")
        base_context = retriever.retrieve_raw(query, use_rerank=False)
        logger.info("基础检索摘要:")
        for idx, item in enumerate(base_context, 1):
            logger.info(f"{idx}. {item.get('summary', '')}（{item.get('score', '')}）")
        
        # 测试重排序检索
        if RAG_CONFIG["reranker"].get("enable", False):
            logger.info("\n--- 重排序结果 ---")
            start_time = time.time()
            rerank_context = retriever.retrieve_raw(query, use_rerank=True)
            end_time = time.time()
            logger.info("重排序摘要:")
            for idx, item in enumerate(rerank_context, 1):
                logger.info(f"{idx}. {item.get('summary', '')}（{item.get('score', '')}）")
            logger.info(f"重排耗时: {end_time - start_time:.2f}秒")
        else:
            logger.warning("重排序功能未启用")

def test_ollama_service():
    """测试Ollama服务是否可用"""
    logger.info("\n=== 测试Ollama服务 ===")
    try:
        start_time = time.time()
        response = requests.get(SERVICE_CONFIG["ollama_host"], timeout=5)
        latency = time.time() - start_time
        
        if response.status_code == 200:
            logger.info(f"✅ Ollama服务可用! 响应时间: {latency:.2f}秒")
            return True
        else:
            logger.error(f"❌ Ollama服务异常: HTTP {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ 连接Ollama失败: {str(e)}")
        return False

def test_embedding_generation():
    """测试嵌入生成功能"""
    logger.info("\n=== 测试嵌入生成 ===")
    from RAG.embeddings import EmbeddingModel
    
    try:
        model = EmbeddingModel()
        texts = ["这是一个测试句子"]
        
        start_time = time.time()
        embeddings = model.embed_texts(texts)
        latency = time.time() - start_time
        
        if embeddings and len(embeddings) == len(texts):
            logger.info(f"✅ 嵌入生成成功! 耗时: {latency:.2f}秒")
            logger.info(f"嵌入维度: {len(embeddings[0])}")
            return True
        else:
            logger.error(f"❌ 嵌入生成失败: 返回数量不匹配")
            return False
    except Exception as e:
        logger.error(f"❌ 嵌入生成异常: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("=== 开始RAG重排序诊断 ===")
    
    # 检查基础服务
    ollama_ok = test_ollama_service()
    embedding_ok = test_embedding_generation()
    
    if ollama_ok and embedding_ok:
        # 运行重排序测试
        test_rerank_results()
    else:
        logger.error("基础服务不可用，无法进行重排序测试")