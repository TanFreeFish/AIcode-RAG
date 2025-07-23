import requests
import numpy as np
from typing import List
from config import RAG_CONFIG
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(self):
        config = RAG_CONFIG["embeddings"]
        self.model_type = config["model_type"]
        self.model_name = config["model_name"]
        self.dim = config.get("dim", 384)
        self.api_url = "http://localhost:11434/api/embeddings"
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为嵌入向量"""
        if not texts:
            return []
            
        if self.model_type == "ollama":
            return self._embed_with_ollama(texts)
        elif self.model_type == "huggingface":
            return self._embed_with_huggingface(texts)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _embed_with_ollama(self, texts: List[str]) -> List[List[float]]:
        """使用Ollama API生成嵌入 - 增强错误处理和重试机制"""
        embeddings = []
        for text in texts:
            if not text.strip():
                embeddings.append([])
                continue
                
            for attempt in range(3):  # 最多重试3次
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
                            # 检查嵌入维度
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
                    
                    # 最后一次尝试仍然失败
                    if attempt == 2:
                        logger.error(f"Failed to generate embedding after 3 attempts for text: {text[:50]}...")
                        embeddings.append([])
                except Exception as e:
                    logger.error(f"Ollama embedding error: {str(e)}")
                    if attempt == 2:
                        embeddings.append([])
        return embeddings
    
    def _embed_with_huggingface(self, texts: List[str]) -> List[List[float]]:
        # 实现HuggingFace嵌入逻辑（根据您的需求）
        return [[] for _ in texts]  # 示例返回