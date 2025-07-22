import os
import json
import numpy as np
from config import VECTOR_STORE_DIR, RAG_CONFIG
from pathlib import Path

class VectorStore:
    def __init__(self):
        self.store_dir = Path(VECTOR_STORE_DIR)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        config = RAG_CONFIG["vector_store"]
        self.store_type = config["type"]
        self.index_name = config["index_name"]
        self.index_path = self.store_dir / f"{self.index_name}.json"
        
        self.index = {}
    
    def load_index(self):
        """加载向量索引"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    self.index = json.load(f)
                print(f"Loaded vector index with {len(self.index)} chunks")
            except Exception as e:
                print(f"Error loading vector index: {str(e)}")
                self.index = {}
        else:
            print("No existing vector index found")
            self.index = {}
    
    def save_index(self):
        """保存向量索引"""
        try:
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, ensure_ascii=False, indent=2)
            print(f"Saved vector index with {len(self.index)} chunks")
        except Exception as e:
            print(f"Error saving vector index: {str(e)}")
    
    def add_chunks(self, chunks, embeddings):
        """添加文本块和嵌入到向量存储"""
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:  # 确保嵌入不为空
                chunk_id = chunk["chunk_id"]
                self.index[chunk_id] = {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "embedding": embedding
                }
        self.save_index()
    
    def similarity_search(self, query_embedding, top_k=5):
        """在向量存储中搜索最相似的文本块"""
        if not self.index:
            return []
        
        # 计算相似度得分
        scores = []
        for chunk_id, chunk_data in self.index.items():
            if not chunk_data.get("embedding"):
                continue
            embedding = np.array(chunk_data["embedding"])
            score = self._cosine_similarity(np.array(query_embedding), embedding)
            scores.append((score, chunk_id, chunk_data))
        
        # 按相似度排序并返回前k个结果
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]
    
    def _cosine_similarity(self, a, b):
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))