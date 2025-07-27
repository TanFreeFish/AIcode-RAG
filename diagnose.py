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
    
    # 新增摘要层测试
    logger.info("\n=== 测试语义摘要层 ===")
    test_queries = [
        "人如何自律",
        "html是什么",
        "what is codeAID",
        "荀子有哪些名言"
    ]
    
    for query in test_queries:
        logger.info(f"\n摘要查询: '{query}'")
        # 直接调用向量存储的摘要搜索
        from RAG.vector_store import VectorStore
        vector_store = VectorStore()
        vector_store.load_index()
        
        # 生成查询嵌入
        from RAG.embeddings import EmbeddingModel
        embedding_model = EmbeddingModel()
        query_embedding = embedding_model.embed_texts([query])[0]
        
        if query_embedding:
            # 使用摘要索引搜索
            results = vector_store.summary_index.get_nns_by_vector(
                query_embedding, 
                5,  # 获取前5个结果
                include_distances=True
            )
            
            indices, distances = results
            logger.info(f"找到 {len(indices)} 个相关摘要:")
            for i, (idx, dist) in enumerate(zip(indices, distances)):
                if idx < len(vector_store.metadata):
                    metadata = vector_store.metadata[idx]
                    # 计算余弦相似度
                    cosine_sim = 1 - (dist ** 2) / 2.0
                    
                    # 输出摘要和对应内容
                    logger.info(f"{i+1}. [相似度: {cosine_sim:.3f}] 摘要: '{metadata['summary']}'")
                    logger.info(f"   来源: {metadata['source']}")
                    logger.info(f"   内容: {metadata['text'][:500]}...")  # 输出前200个字符
                    logger.info("\n")
        else:
            logger.warning(f"无法生成查询 '{query}' 的嵌入向量")

if __name__ == "__main__":
    test_rag()