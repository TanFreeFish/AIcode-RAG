from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from config import RAG_CONFIG
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.vector_store.load_index()
        self.top_k = RAG_CONFIG["retriever"]["top_k"]
        self.score_threshold = RAG_CONFIG["retriever"]["score_threshold"]
    
    def retrieve(self, query: str) -> str:
        """检索与查询相关的上下文"""
        # 生成查询嵌入
        query_embedding = self.embedding_model.embed_texts([query])
        if not query_embedding or not query_embedding[0]:
            logger.warning(f"Failed to generate embedding for query: '{query}'")
            return ""
        
        # 确保嵌入向量格式正确
        if not isinstance(query_embedding[0], list) or not all(isinstance(x, float) for x in query_embedding[0]):
            logger.error(f"Invalid embedding format for query: '{query}'")
            return ""
        
        # 执行相似度搜索
        results = self.vector_store.similarity_search(
            query_embedding[0], 
            top_k=self.top_k
        )
        
        # 构建上下文
        context = []
        for score, chunk_id, chunk_data in results:
            if score >= self.score_threshold:
                context.append({
                    "text": chunk_data["text"],
                    "source": chunk_data["source"],
                    "score": round(score, 3)
                })
        
        # 格式化上下文
        return self._format_context(context)
    
    def _format_context(self, context_items) -> str:
        """格式化检索到的上下文"""
        if not context_items:
            return "没有找到相关上下文信息"
        
        context_str = "检索到的相关上下文信息：\n\n"
        for i, item in enumerate(context_items, 1):
            source_name = Path(item["source"]).name
            context_str += f"### 上下文片段 {i} (来源: {source_name}, 相似度: {item['score']})\n"
            context_str += f"{item['text']}\n\n"
        
        return context_str.strip()