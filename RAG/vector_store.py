# RAG/vector_store.py
from annoy import AnnoyIndex
import json
import numpy as np
import os
from pathlib import Path
from config import VECTOR_STORE_DIR, RAG_CONFIG
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.store_dir = Path(VECTOR_STORE_DIR)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        config = RAG_CONFIG["vector_store"]
        self.store_type = config["type"]
        self.index_name = config["index_name"]
        self.index_path = self.store_dir / f"{self.index_name}.ann"
        self.metadata_path = self.store_dir / f"{self.index_name}_metadata.json"
        self.index = None
        self.metadata = []
        self.chunk_ids = []  # 新增：存储块ID列表
        
        # 从嵌入配置获取维度
        embedding_config = RAG_CONFIG["embeddings"]
        self.dim = embedding_config.get("dim", 384)  # 从配置获取维度
        
        self.distance_metric = "angular"  # 余弦相似度
        
        # 确保索引对象被正确创建
        self.load_index()
        
        # 双重检查索引对象
        if self.index is None:
            logger.warning("Index was None after load_index(), creating new index")
            self.index = AnnoyIndex(self.dim, self.distance_metric)

    def load_index(self):
        try:
            if self.index_path.exists():
                logger.info(f"Loading existing index from {self.index_path}")
                self.index = AnnoyIndex(self.dim, self.distance_metric)
                self.index.load(str(self.index_path))
                
                if self.metadata_path.exists():
                    with open(self.metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    # 修复：正确加载元数据和块ID
                    self.metadata = metadata.get("chunks", [])
                    self.chunk_ids = metadata.get("chunk_ids", [])
                logger.info(f"Loaded Annoy index with {len(self.metadata)} chunks")
            else:
                logger.info("No existing index found, creating new index")
                self.index = AnnoyIndex(self.dim, self.distance_metric)
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            # 创建新的索引作为回退
            self.index = AnnoyIndex(self.dim, self.distance_metric)

    def add_chunks(self, chunks, embeddings):
        if not embeddings:
            logger.warning("No embeddings provided, skipping add_chunks")
            return
            
        # 确保索引对象存在
        if self.index is None:
            logger.warning("Index is None, creating new index")
            self.index = AnnoyIndex(self.dim, self.distance_metric)
        
        # 重置元数据和块ID
        self.metadata = []
        self.chunk_ids = []
        
        # 添加向量到 Annoy 索引
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not embedding or len(embedding) == 0:
                logger.warning(f"Skipping empty embedding for chunk {i}")
                continue
                
            # 确保嵌入是浮点数列表
            embedding = [float(x) for x in embedding]
                
            self.index.add_item(i, embedding)
            self.metadata.append({
                "text": chunk["text"],
                "source": chunk["source"]
            })
            self.chunk_ids.append(chunk["chunk_id"])
        
        # 构建索引
        logger.info("Building index...")
        self.index.build(10)  # 默认 10 棵树
        logger.info("Index built successfully")
        
        # 保存索引
        self.save_index()

    def similarity_search(self, query_embedding, top_k=5):
        if self.index is None:
            logger.warning("Index is None, cannot perform search")
            return []
            
        if not query_embedding:
            logger.warning("Empty query embedding")
            return []
            
        # 确保查询嵌入是浮点数列表
        query_embedding = [float(x) for x in query_embedding]
        
        # 获取最近邻
        try:
            indices, distances = self.index.get_nns_by_vector(
                query_embedding, 
                top_k, 
                include_distances=True
            )
        except Exception as e:
            logger.error(f"Error in similarity_search: {str(e)}")
            return []
            
        results = []
        for idx, distance in zip(indices, distances):
            if idx < len(self.metadata):
                # 修复：使用索引直接获取元数据
                chunk_data = self.metadata[idx]
                # 修复：余弦距离转相似度 (1 - distance)
                similarity = 1 - distance
                results.append((similarity, self.chunk_ids[idx], chunk_data))
        
        return sorted(results, key=lambda x: x[0], reverse=True)

    def save_index(self):
        if self.index is None:
            logger.warning("Index is None, cannot save")
            return
            
        self.index.save(str(self.index_path))
        # 修复：正确保存元数据和块ID
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": self.metadata,
                "chunk_ids": self.chunk_ids
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Annoy index with {len(self.metadata)} chunks")