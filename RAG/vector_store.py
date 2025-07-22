# RAG/vector_store.py
from annoy import AnnoyIndex

import json
import numpy as np
import os
from config import VECTOR_STORE_DIR, RAG_CONFIG
from pathlib import Path

class VectorStore:
    def __init__(self):
        self.store_dir = Path(VECTOR_STORE_DIR)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        config = RAG_CONFIG["vector_store"]
        self.store_type = config["type"]  # 保留配置字段（可选）
        self.index_name = config["index_name"]
        self.index_path = self.store_dir / f"{self.index_name}.ann"  # 改为 .ann 后缀
        self.metadata_path = self.store_dir / f"{self.index_name}_metadata.json"
        self.index = None
        self.metadata = {}
        self.dim = 768  # 假设嵌入维度为 768（根据实际模型调整）
        self.distance_metric = "angular"  # 余弦相似度（Annoy 支持）
    def load_index(self):
        if self.index_path.exists():
            self.index = AnnoyIndex(self.dim, self.distance_metric)
            self.index.load(str(self.index_path))
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            print(f"Loaded Annoy index with {len(self.metadata)} chunks")
        else:
            self.index = AnnoyIndex(self.dim, self.distance_metric)
            print("No existing Annoy index found")
    def add_chunks(self, chunks, embeddings):
        if not embeddings:
            return
        # 确保嵌入是 numpy 数组
        embeddings = np.array(embeddings).astype('float32')
        # 添加向量到 Annoy 索引
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = chunk["chunk_id"]
            self.index.add_item(i, embedding.tolist())
            self.metadata[chunk_id] = {
                "text": chunk["text"],
                "source": chunk["source"],
                "embedding": embedding.tolist()
            }
        # 构建索引（n_trees 控制树的数量，影响精度和速度）
        self.index.build(10)  # 默认 10 棵树
        self.save_index()
    def similarity_search(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding]).astype('float32')
        # 获取最近邻（返回索引列表）
        indices, distances = self.index.get_nns_by_vector(query_embedding[0], top_k, include_distances=True)
        results = []
        for idx, distance in zip(indices, distances):
            if idx < len(self.metadata):
                chunk_ids = list(self.metadata.keys())
                chunk_id = chunk_ids[idx]
                chunk_data = self.metadata[chunk_id]
                # Annoy 的余弦相似度范围是 [0, 1]，0 表示完全不相似
                score = 1 - distance  # 转换为相似度得分
                results.append((score, chunk_id, chunk_data))
        return sorted(results, key=lambda x: x[0], reverse=True)
    def save_index(self):
        self.index.save(str(self.index_path))
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved Annoy index with {len(self.metadata)} chunks")