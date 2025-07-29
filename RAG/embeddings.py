import requests
import numpy as np
from typing import List
from config import RAG_CONFIG
import logging
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self):
        """
        @brief 初始化嵌入模型
        """
        config = RAG_CONFIG["embeddings"]
        self.model_type = config["model_type"]
        self.model_name = config["model_name"]
        self.dim = config.get("dim", 384)
        self.api_url = "http://localhost:11434/api/embeddings"
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        @brief 根据配置的模型类型，使用相应的方法将输入文本转换为向量表示
        
        @param texts (List[str]): 需要转换为向量的文本列表
        
        @return List[List[float]]: 对应的向量表示列表，每个向量是一个浮点数列表
        """
        if not texts:
            return []
            
        if self.model_type == "ollama":
            return self._embed_with_ollama(texts)
        elif self.model_type == "huggingface":
            return self._embed_with_huggingface(texts)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _embed_with_ollama(self, texts: List[str]) -> List[List[float]]:
        """
        @brief 通过HTTP请求调用Ollama的嵌入API，将文本转换为向量表示，并包含重试机制
        
        @param texts (List[str]): 需要生成嵌入的文本列表
        
        @return List[List[float]]: 文本对应的向量表示列表
        """
        embeddings = []
        for text in texts:
            if not text.strip():
                embeddings.append([])
                continue
                
            for attempt in range(3):  
                try:
                    response = requests.post(
                        self.api_url,
                        json={
                            "model": self.model_name,
                            "prompt": text
                        },
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if "embedding" in data and data["embedding"]:
                            embedding = data["embedding"]
                            
                            if len(embedding) == self.dim:
                                embeddings.append(embedding)
                                break
                            else:
                                logger.warning(f"Embedding dimension mismatch: expected {self.dim}, got {len(embedding)}")
                                embeddings.append([])
                                break
                        else:
                            logger.warning(f"Empty embedding for text: {text[:50]}...")
                    else:
                        logger.error(f"Error embedding text: {response.status_code} - {response.text}")
                    
                   
                    if attempt == 2:
                        logger.error(f"Failed to generate embedding after 3 attempts for text: {text[:50]}...")
                        embeddings.append([])
                except Exception as e:
                    logger.error(f"Ollama embedding error: {str(e)}")
                    if attempt == 2:
                        embeddings.append([])
        return embeddings
    
    def _embed_with_huggingface(self, texts: List[str]) -> List[List[float]]:
        """
        @brief 占位方法，用于HuggingFace模型的嵌入生成（当前未实现）
        
        @param texts (List[str]): 需要生成嵌入的文本列表
        
        @return List[List[float]]: 空向量列表，每个元素都是空列表
        """
        return [[] for _ in texts]