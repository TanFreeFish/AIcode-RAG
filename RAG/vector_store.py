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
        self.chunk_ids = []  # 存储块ID列表
        
        # 从嵌入配置获取维度
        embedding_config = RAG_CONFIG["embeddings"]
        self.dim = embedding_config.get("dim", 384)
        
        self.distance_metric = "angular"  # 余弦相似度
        
        # 确保索引对象被正确创建
        self.load_index()
        
        # 双重检查索引对象
        if self.index is None:
            logger.warning("Index was None after load_index(), creating new index")
            self.index = AnnoyIndex(self.dim, self.distance_metric)

    def load_index(self):
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                logger.info(f"Loading existing index from {self.index_path}")
                self.index = AnnoyIndex(self.dim, self.distance_metric)
                self.index.load(str(self.index_path))
                
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                # 正确加载元数据和块ID
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
        
        # 添加向量到Annoy索引
        valid_count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if not embedding or len(embedding) != self.dim:
                logger.warning(f"Skipping invalid embedding for chunk {i}")
                continue
                
            # 确保嵌入是浮点数列表
            try:
                embedding = [float(x) for x in embedding]
                self.index.add_item(i, embedding)
                self.metadata.append({
                    "text": chunk["text"],
                    "source": chunk["source"]
                })
                self.chunk_ids.append(chunk["chunk_id"])
                valid_count += 1
            except Exception as e:
                logger.error(f"Error adding chunk {i}: {str(e)}")
        
        if valid_count == 0:
            logger.error("No valid embeddings added to index")
            return
            
        # 构建索引
        logger.info(f"Building index with {valid_count} items...")
        self.index.build(10)  # 默认10棵树
        logger.info("Index built successfully")
        
        # 保存索引
        self.save_index()

    def similarity_search(self, query_embedding, top_k=5):
        if self.index is None:
            logger.warning("Index is None, cannot perform search")
            return []
            
        if not query_embedding or len(query_embedding) != self.dim:
            logger.error(f"Invalid query embedding: expected dim={self.dim}, got {len(query_embedding) if query_embedding else 'none'}")
            return []
            
        # 确保查询嵌入是浮点数列表
        try:
            query_embedding = [float(x) for x in query_embedding]
        except Exception as e:
            logger.error(f"Error converting query embedding: {str(e)}")
            return []
        
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
            if idx < len(self.metadata) and idx < len(self.chunk_ids):
                chunk_data = self.metadata[idx]
                # 余弦距离转相似度 (1 - distance)
                similarity = 1 - distance
                results.append((similarity, self.chunk_ids[idx], chunk_data))
        
        return sorted(results, key=lambda x: x[0], reverse=True)

    def save_index(self):
        if self.index is None:
            logger.warning("Index is None, cannot save")
            return
            
        self.index.save(str(self.index_path))
        # 保存元数据和块ID
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": self.metadata,
                "chunk_ids": self.chunk_ids
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Annoy index with {len(self.metadata)} chunks")