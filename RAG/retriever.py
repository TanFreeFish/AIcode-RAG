from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from config import RAG_CONFIG, SERVICE_CONFIG
from pathlib import Path
import logging
import json
import requests
import re  


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self):
        """
        @brief 初始化检索器
        """
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()
        self.vector_store.load_index()
        
        
        retriever_config = RAG_CONFIG["retriever"]
        self.top_k = retriever_config["top_k"]
        self.score_threshold = retriever_config["score_threshold"]
        self.enable_rerank = retriever_config["enable_rerank"]
        
        
        reranker_config = RAG_CONFIG.get("reranker", {})
        self.reranker_enable = reranker_config.get("enable", False)
        self.reranker_model = reranker_config.get("model_name", "qwen:7b")
        self.top_n_for_rerank = reranker_config.get("top_n_for_rerank", 10)
        self.rerank_score_threshold = reranker_config.get("score_threshold", 0.3)
        self.reranker_prompt_template = reranker_config.get("prompt_template", "")
        self.ollama_host = SERVICE_CONFIG["ollama_host"]
        self.rerank_timeout = SERVICE_CONFIG["rerank_timeout"]
    
    def retrieve(self, query: str, use_rerank: bool = None) -> str:
        """
        @brief 根据输入查询检索相关文档片段，可选择是否进行重排序，并返回格式化的上下文字符串
        
        @param query (str): 用户的查询字符串
        @param use_rerank (bool, optional): 是否启用重排序功能，默认为None时使用配置值
        
        @return str: 格式化的上下文信息字符串
        """
        
        if use_rerank is None:
            use_rerank = self.reranker_enable or self.enable_rerank
            
        
        query_embedding = self.embedding_model.embed_texts([query])
        if not query_embedding or not query_embedding[0]:
            logger.warning(f"Failed to generate embedding for query: '{query}'")
            return ""
        
        
        if not isinstance(query_embedding[0], list) or not all(isinstance(x, float) for x in query_embedding[0]):
            logger.error(f"Invalid embedding format for query: '{query}'")
            return ""
        
        
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
        
        
        return self._format_context(context)
    
    def retrieve_raw(self, query: str, use_rerank: bool = None) -> list:
        """
        @brief 根据输入查询检索相关文档片段，返回原始数据结构
        
        @param query (str): 用户的查询字符串
        @param use_rerank (bool, optional): 是否启用重排序功能，默认为None时使用配置值
        
        @return list: 包含检索结果的列表，每个元素是包含文本、摘要等信息的字典
        """
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
        """
        @brief 通过调用大语言模型对初步检索结果进行相关性重排序
        
        @param query (str): 用户的原始查询
        @param results (list): 初步检索结果列表
        
        @return list: 重排序后的结果列表，格式与输入相同
        """
        if len(results) < 2:  
            return None
        
        
        summaries = []
        for i, (score, _, chunk_data) in enumerate(results[:self.top_n_for_rerank]):
            summaries.append(f"[{i+1}] {chunk_data['summary']}")
        
        
        prompt = self.reranker_prompt_template.format(
            query=query,
            summaries="\n".join(summaries)
        )
        
        try:
            
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
                
                response_text = response.json().get("response", "").strip()
                logger.info(f"大模型原始返回: {response_text}")
                ranked_pairs = self._parse_rerank_response(response_text)
                
                if not ranked_pairs:
                    logger.info("重排序未返回有效结果")
                    return None
                
                
                reranked = []
                for idx, score in ranked_pairs:
                    
                    if 1 <= idx <= len(summaries):
                        
                        original_index = idx - 1  
                        reranked.append((
                            results[original_index][0],  
                            score,                      
                            results[original_index][1],  
                            results[original_index][2]   
                        ))
                
                
                reranked.sort(key=lambda x: x[1], reverse=True)
                
                
                final_results = []
                for item in reranked:
                    
                    final_results.append((item[0], item[2], item[3]))
                
                
                if self.top_n_for_rerank < len(results):
                    final_results.extend(results[self.top_n_for_rerank:])
                
                return final_results
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
        
        return None
    
    def _parse_rerank_response(self, response: str) -> list:
        """
        @brief 解析大语言模型返回的重排序结果，提取索引-分数对
        
        @param response (str): 大语言模型的原始响应字符串
        
        @return list: 解析后的索引-分数对列表，每个元素是(索引, 分数)元组
        """
        try:
            
            json_start = response.find('[')
            json_end = response.rfind(']')
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end+1]
                data = json.loads(json_str)
                
                
                if isinstance(data, list):
                    valid_pairs = []
                    for item in data:
                        
                        if (isinstance(item, list) and len(item) == 2):
                            idx = item[0]
                            score = item[1]
                            
                            if (isinstance(idx, int) and 
                                (isinstance(score, float) or isinstance(score, int)) and
                                score > self.rerank_score_threshold):
                                valid_pairs.append((idx, float(score)))
                    return valid_pairs
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"JSON解析失败: {str(e)}，尝试其他格式")
        
        
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
        
        
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs
    
    def _format_context(self, context_items) -> str:
        """
        @brief 将检索到的上下文项格式化为可读的字符串格式
        
        @param context_items (list): 上下文项列表，每个元素包含文本、摘要等信息
        
        @return str: 格式化后的上下文字符串
        """
        if not context_items:
            return "没有找到相关上下文信息"
        
        context_str = "检索到的相关上下文信息：\n\n"
        for i, item in enumerate(context_items, 1):
            source_name = Path(item["source"]).name
            context_str += f"===上下文片段 {i} (来源: {source_name}, 相似度: {item['score']}, 摘要: {item['summary']})\n"
            context_str += f"{item['text']}\n\n"
        
        return context_str.strip()