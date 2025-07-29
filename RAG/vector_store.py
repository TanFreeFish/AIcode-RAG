
from annoy import AnnoyIndex
import json
import numpy as np
import os
from pathlib import Path
from config import VECTOR_STORE_DIR, RAG_CONFIG
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, rebuild_mode=False):
        """
        @brief 初始化向量存储
        
        @param rebuild_mode (bool): 是否重建模式
        """
        self.store_dir = Path(VECTOR_STORE_DIR)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        config = RAG_CONFIG["vector_store"]
        self.store_type = config["type"]
        self.index_name = config["index_name"]
        self.index_path = self.store_dir / f"{self.index_name}.ann"
        self.summary_index_path = self.store_dir / f"{self.index_name}_summary.ann"
        self.metadata_path = self.store_dir / f"{self.index_name}_metadata.json"
        self.index = None
        self.summary_index = None
        self.metadata = []
        self.chunk_ids = []
        self.rebuild_mode = rebuild_mode
        
        
        embedding_config = RAG_CONFIG["embeddings"]
        self.dim = embedding_config.get("dim", 384)
        
        self.distance_metric = "angular"
        
        
        self.load_index()
        
    def load_index(self):
        """
        @brief 根据是否存在已保存的索引文件决定是加载现有索引还是创建新索引
        """
        try:
            
            if self.rebuild_mode or not (self.index_path.exists() and self.summary_index_path.exists() and self.metadata_path.exists()):
                logger.info("Creating new index (rebuild mode or no existing index)")
                self.index = AnnoyIndex(self.dim, self.distance_metric)
                self.summary_index = AnnoyIndex(self.dim, self.distance_metric)
                self.metadata = []
                self.chunk_ids = []
            else:
                logger.info(f"Loading existing index from {self.index_path}")
                self.index = AnnoyIndex(self.dim, self.distance_metric)
                self.index.load(str(self.index_path))
                
                
                self.summary_index = AnnoyIndex(self.dim, self.distance_metric)
                self.summary_index.load(str(self.summary_index_path))
                
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                self.metadata = metadata.get("chunks", [])
                self.chunk_ids = metadata.get("chunk_ids", [])
                logger.info(f"Loaded Annoy index with {len(self.metadata)} chunks")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            
            self.index = AnnoyIndex(self.dim, self.distance_metric)
            self.summary_index = AnnoyIndex(self.dim, self.distance_metric)
            self.metadata = []
            self.chunk_ids = []
    
    def add_chunks(self, chunks, embeddings, progress_callback=None):
        """
        @brief 将文本块及其对应的向量表示添加到Annoy索引中，并保存元数据
        
        @param chunks (list): 文本块列表，每个元素包含文本、摘要等信息
        @param embeddings (list): 向量表示列表，与文本块一一对应
        @param progress_callback (function, optional): 进度回调函数，用于报告处理进度
        
        @return bool: 添加成功返回True，否则返回False
        """
        if not embeddings:
            logger.warning("No embeddings provided, skipping add_chunks")
            return False
            
        
        if self.index is None:
            logger.warning("Index is None, creating new index")
            self.index = AnnoyIndex(self.dim, self.distance_metric)
        if self.summary_index is None:
            logger.warning("Summary index is None, creating new summary index")
            self.summary_index = AnnoyIndex(self.dim, self.distance_metric)
        
        
        progress_callback = progress_callback or (lambda **kw: None)
        total_chunks = len(chunks)
        
        
        progress_callback(
            stage="index",
            total=total_chunks,
            current=0,
            message="开始构建索引",
            details=f"共 {total_chunks} 个文本块"
        )
        
        
        self.metadata = []
        self.chunk_ids = []
        
        
        valid_count = 0
        for i, (chunk, embedding) in enumerate(tqdm(zip(chunks, embeddings), desc="构建索引")):
            if not embedding or len(embedding) != self.dim:
                logger.warning(f"Skipping invalid embedding for chunk {i}")
                continue
                
            try:
                
                embedding = [float(x) for x in embedding]
                
                
                embedding_arr = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(embedding_arr)
                if norm > 0:
                    embedding_arr = embedding_arr / norm
                else:
                    
                    logger.warning(f"Zero vector embedding for chunk {i}, skipping")
                    continue
                
                
                self.index.add_item(valid_count, embedding_arr)
                
                
                summary_embedding = self._get_summary_embedding(chunk["summary"])
                if summary_embedding and len(summary_embedding) == self.dim:
                    summary_arr = np.array(summary_embedding, dtype=np.float32)
                    summary_norm = np.linalg.norm(summary_arr)
                    if summary_norm > 0:
                        summary_arr = summary_arr / summary_norm
                    self.summary_index.add_item(valid_count, summary_arr)
                else:
                    
                    self.summary_index.add_item(valid_count, embedding_arr)
                
                self.metadata.append({
                    "text": chunk["text"],
                    "summary": chunk["summary"],
                    "source": chunk["source"]
                })
                self.chunk_ids.append(chunk["chunk_id"])
                valid_count += 1
                
                
                if (i + 1) % 10 == 0 or (i + 1) == total_chunks:
                    progress_callback(
                        stage="index",
                        current=i + 1,
                        total=total_chunks,
                        message=f"正在添加文本块 {i+1}/{total_chunks}",
                        details=f"有效块: {valid_count}"
                    )
            except Exception as e:
                logger.error(f"Error adding chunk {i}: {str(e)}")
        
        if valid_count == 0:
            logger.error("No valid embeddings added to index")
            progress_callback(
                stage="index",
                message="未添加有效嵌入",
                status="error"
            )
            return False
            
        
        logger.info(f"Building index with {valid_count} items...")
        progress_callback(
            stage="index",
            message="正在构建索引结构...",
            details=f"共 {valid_count} 个项目"
        )
        
        try:
            self.index.build(10)
            self.summary_index.build(10)
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            progress_callback(
                stage="index",
                message="索引构建失败",
                status="error"
            )
            return False
        
        logger.info("Index built successfully")
        progress_callback(
            stage="index",
            message="索引构建完成",
            status="completed"
        )
        
        
        self.save_index()
        return True

    def _get_summary_embedding(self, summary: str) -> list:
        """
        @brief 为输入的摘要文本生成对应的向量表示
        
        @param summary (str): 需要生成向量的摘要文本
        
        @return list: 摘要的向量表示，失败时返回None
        """
        try:
            
            from RAG.embeddings import EmbeddingModel
            embedding_model = EmbeddingModel()
            if summary and summary.strip():
                return embedding_model.embed_texts([summary])[0]
            return None
        except ImportError as e:
            logger.error(f"导入EmbeddingModel失败: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"生成摘要嵌入失败: {str(e)}")
            return None

    def similarity_search(self, query_embedding, top_k=5):
        """
        @brief 在向量索引中查找与查询向量最相似的文本块
        
        @param query_embedding (list): 查询文本的向量表示
        @param top_k (int): 返回最相似结果的数量，默认为5
        
        @return list: 相似度搜索结果列表，每个元素包含相似度分数、块ID和元数据
        """
        if self.index is None or self.summary_index is None:
            logger.warning("Index is None, cannot perform search")
            return []
            
        if not query_embedding or len(query_embedding) != self.dim:
            logger.error(f"Invalid query embedding: expected dim={self.dim}, got {len(query_embedding) if query_embedding else 'none'}")
            return []
            
        
        try:
            query_embedding = [float(x) for x in query_embedding]
        except Exception as e:
            logger.error(f"Error converting query embedding: {str(e)}")
            return []
        
        
        query_embedding_arr = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding_arr)
        if query_norm > 0:
            query_embedding_arr = query_embedding_arr / query_norm
        else:
            
            query_embedding_arr = np.zeros(self.dim, dtype=np.float32)
        
        
        try:
            summary_indices, summary_distances = self.summary_index.get_nns_by_vector(
                query_embedding_arr, 
                top_k * 3,  
                include_distances=True,
                search_k=-1
            )
        except Exception as e:
            logger.error(f"Error in summary similarity search: {str(e)}")
            summary_indices, summary_distances = [], []
        
        
        results = []
        for idx, angular_dist in zip(summary_indices, summary_distances):
            if idx < len(self.metadata) and idx < len(self.chunk_ids):
                
                cosine_sim = 1 - (angular_dist ** 2) / 2.0
                cosine_sim = max(-1.0, min(1.0, cosine_sim))
                
                
                try:
                    item_vector = self.index.get_item_vector(idx)
                    main_cosine_sim = np.dot(query_embedding_arr, item_vector)
                    main_cosine_sim = max(-1.0, min(1.0, main_cosine_sim))
                except:
                    main_cosine_sim = cosine_sim
                
                
                results.append((
                    main_cosine_sim,  
                    self.chunk_ids[idx], 
                    self.metadata[idx],
                    cosine_sim  
                ))
        
        
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
        return [(score, chunk_id, data) for score, chunk_id, data, _ in sorted_results]
    
    def save_index(self):
        """
        @brief 将当前的向量索引和元数据保存到磁盘文件中
        """
        if self.index is None or self.summary_index is None:
            logger.warning("Index is None, cannot save")
            return
            
        self.index.save(str(self.index_path))
        self.summary_index.save(str(self.summary_index_path))
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": self.metadata,
                "chunk_ids": self.chunk_ids
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Annoy index with {len(self.metadata)} chunks")