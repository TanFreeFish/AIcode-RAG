import requests
import numpy as np
from typing import List
from config import RAG_CONFIG

class EmbeddingModel:
    def __init__(self):
        config = RAG_CONFIG["embeddings"]
        self.model_type = config["model_type"]
        self.model_name = config["model_name"]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为嵌入向量"""
        if self.model_type == "ollama":
            return self._embed_with_ollama(texts)
        elif self.model_type == "huggingface":
            return self._embed_with_huggingface(texts)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _embed_with_ollama(self, texts: List[str]) -> List[List[float]]:
        """使用Ollama API生成嵌入"""
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    "http://localhost:11434/api/embeddings",
                    json={
                        "model": self.model_name,
                        "prompt": text
                    }
                )
                if response.status_code == 200:
                    embeddings.append(response.json().get("embedding", []))
                else:
                    print(f"Error embedding text: {response.text}")
                    embeddings.append([])
            except Exception as e:
                print(f"Ollama embedding error: {str(e)}")
                embeddings.append([])
        
        return embeddings
    
    def _embed_with_huggingface(self, texts: List[str]) -> List[List[float]]:
        """使用Hugging Face模型生成嵌入（占位符）"""
        # 实际项目中应实现Hugging Face嵌入
        print("Hugging Face embeddings not implemented yet")
        return [np.random.rand(768).tolist() for _ in texts]