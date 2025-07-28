from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from config import RAG_CONFIG, SERVICE_CONFIG
from pathlib import Path
import logging
import json
import requests
import re  # 新增导入正则模块

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.vector_store.load_index()
        
        # 检索器配置
        retriever_config = RAG_CONFIG["retriever"]
        self.top_k = retriever_config["top_k"]
        self.score_threshold = retriever_config["score_threshold"]
        self.enable_rerank = retriever_config["enable_rerank"]
        
        # 重排序配置
        reranker_config = RAG_CONFIG.get("reranker", {})
        self.reranker_enable = reranker_config.get("enable", False)
        self.reranker_model = reranker_config.get("model_name", "qwen:7b")
        self.top_n_for_rerank = reranker_config.get("top_n_for_rerank", 10)
        self.rerank_score_threshold = reranker_config.get("score_threshold", 0.3)
        self.reranker_prompt_template = reranker_config.get("prompt_template", "")
        self.ollama_host = SERVICE_CONFIG["ollama_host"]
        self.rerank_timeout = SERVICE_CONFIG["rerank_timeout"]
    
    def retrieve(self, query: str, use_rerank: bool = None) -> str:
        """检索与查询相关的上下文，可选二次重排序"""
        # 确定是否使用重排序
        if use_rerank is None:
            use_rerank = self.reranker_enable or self.enable_rerank
            
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
        
        # 二次重排序
        if use_rerank and results:
            reranked_results = self._rerank_documents(query, results)
            if reranked_results:
                results = reranked_results
        
        # 构建上下文
        context = []
        for score, chunk_id, chunk_data in results:
            if score >= self.score_threshold:
                context.append({
                    "text": chunk_data["text"],
                    "summary": chunk_data["summary"],
                    "source": chunk_data["source"],
                    "score": round(score, 3)
                })
        
        # 格式化上下文
        return self._format_context(context)
    
    def retrieve_raw(self, query: str, use_rerank: bool = None) -> list:
        """检索与查询相关的上下文，返回原始结构（list）"""
        if use_rerank is None:
            use_rerank = self.reranker_enable or self.enable_rerank
        query_embedding = self.embedding_model.embed_texts([query])
        if not query_embedding or not query_embedding[0]:
            logger.warning(f"Failed to generate embedding for query: '{query}'")
            return []
        if not isinstance(query_embedding[0], list) or not all(isinstance(x, float) for x in query_embedding[0]):
            logger.error(f"Invalid embedding format for query: '{query}'")
            return []
        results = self.vector_store.similarity_search(
            query_embedding[0],
            top_k=self.top_k
        )
        if use_rerank and results:
            reranked_results = self._rerank_documents(query, results)
            if reranked_results:
                results = reranked_results
        context = []
        for score, chunk_id, chunk_data in results:
            if score >= self.score_threshold:
                context.append({
                    "text": chunk_data["text"],
                    "summary": chunk_data["summary"],
                    "source": chunk_data["source"],
                    "score": round(score, 3)
                })
        return context
    
    def _rerank_documents(self, query: str, results: list) -> list:
        """使用大模型对文档进行二元组重排序"""
        if len(results) < 2:  # 文档太少无需重排序
            return None
        
        # 提取摘要用于重排序
        summaries = []
        for i, (score, _, chunk_data) in enumerate(results[:self.top_n_for_rerank]):
            summaries.append(f"[{i+1}] {chunk_data['summary']}")
        
        # 构建提示词
        prompt = self.reranker_prompt_template.format(
            query=query,
            summaries="\n".join(summaries)
        )
        
        try:
            # 调用大模型进行重排序
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.reranker_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=self.rerank_timeout
            )
            
            if response.status_code == 200:
                # 解析模型响应
                response_text = response.json().get("response", "").strip()
                logger.info(f"大模型原始返回: {response_text}")
                ranked_pairs = self._parse_rerank_response(response_text)
                
                if not ranked_pairs:
                    logger.info("重排序未返回有效结果")
                    return None
                
                # 应用重排序
                reranked = []
                for idx, score in ranked_pairs:
                    # 检查索引是否在有效范围内
                    if 1 <= idx <= len(summaries):
                        # 保留原始结果并添加重排序分数
                        original_index = idx - 1  # 转换为0-based索引
                        reranked.append((
                            results[original_index][0],  # 原始相似度分数
                            score,                      # 重排序相关性分数
                            results[original_index][1],  # chunk_id
                            results[original_index][2]   # chunk_data
                        ))
                
                # 按重排序分数降序排列
                reranked.sort(key=lambda x: x[1], reverse=True)
                
                # 组合结果：重排序部分 + 未重排序部分
                final_results = []
                for item in reranked:
                    # 只保留原始数据结构
                    final_results.append((item[0], item[2], item[3]))
                
                # 添加未参与重排序的结果
                if self.top_n_for_rerank < len(results):
                    final_results.extend(results[self.top_n_for_rerank:])
                
                return final_results
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
        
        return None
    
    def _parse_rerank_response(self, response: str) -> list:
        """解析大模型的重排序响应（二元组格式）"""
        try:
            # 尝试提取JSON数组部分
            json_start = response.find('[')
            json_end = response.rfind(']')
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end+1]
                data = json.loads(json_str)
                
                # 验证数据格式
                if isinstance(data, list):
                    valid_pairs = []
                    for item in data:
                        # 检查是否为二元组 [index, score]
                        if (isinstance(item, list) and len(item) == 2):
                            idx = item[0]
                            score = item[1]
                            # 验证类型和阈值
                            if (isinstance(idx, int) and 
                                (isinstance(score, float) or isinstance(score, int)) and
                                score > self.rerank_score_threshold):
                                valid_pairs.append((idx, float(score)))
                    return valid_pairs
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"JSON解析失败: {str(e)}，尝试其他格式")
        
        # 备用解析：正则匹配二元组
        pattern = r'\[(\d+)\s*,\s*([0-9]*\.?[0-9]+)\]'
        pairs = []
        for match in re.findall(pattern, response):
            try:
                idx = int(match[0])
                score = float(match[1])
                if score > self.rerank_score_threshold:
                    pairs.append((idx, score))
            except ValueError:
                continue
        
        # 按分数降序排序
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
    
    def _format_context(self, context_items) -> str:
        """格式化检索到的上下文"""
        if not context_items:
            return "没有找到相关上下文信息"
        
        context_str = "检索到的相关上下文信息：\n\n"
        for i, item in enumerate(context_items, 1):
            source_name = Path(item["source"]).name
            context_str += f"### 上下文片段 {i} (来源: {source_name}, 相似度: {item['score']}, 摘要: {item['summary']})\n"
            context_str += f"{item['text']}\n\n"
        
        return context_str.strip()