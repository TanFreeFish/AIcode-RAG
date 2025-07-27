# RAG/vector_store.py
from annoy import AnnoyIndex
import json
import numpy as np
import os
from pathlib import Path
from config import VECTOR_STORE_DIR, RAG_CONFIG
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, rebuild_mode=False):
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
        
        # 从嵌入配置获取维度
        embedding_config = RAG_CONFIG["embeddings"]
        self.dim = embedding_config.get("dim", 384)
        
        self.distance_metric = "angular"
        
        # 确保索引对象被正确创建
        self.load_index()
        
    def load_index(self):
        try:
            # 重建模式或索引不存在时创建新索引
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
                
                # 加载摘要索引
                self.summary_index = AnnoyIndex(self.dim, self.distance_metric)
                self.summary_index.load(str(self.summary_index_path))
                
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                # 正确加载元数据和块ID
                self.metadata = metadata.get("chunks", [])
                self.chunk_ids = metadata.get("chunk_ids", [])
                logger.info(f"Loaded Annoy index with {len(self.metadata)} chunks")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            # 创建新的索引作为回退
            self.index = AnnoyIndex(self.dim, self.distance_metric)
            self.summary_index = AnnoyIndex(self.dim, self.distance_metric)
            self.metadata = []
            self.chunk_ids = []
    def add_chunks(self, chunks, embeddings, progress_callback=None):
        if not embeddings:
            logger.warning("No embeddings provided, skipping add_chunks")
            return False
            
        # 确保索引对象存在
        if self.index is None:
            logger.warning("Index is None, creating new index")
            self.index = AnnoyIndex(self.dim, self.distance_metric)
        if self.summary_index is None:
            logger.warning("Summary index is None, creating new summary index")
            self.summary_index = AnnoyIndex(self.dim, self.distance_metric)
        
        # 设置进度回调
        progress_callback = progress_callback or (lambda **kw: None)
        total_chunks = len(chunks)
        
        # 发送进度开始消息
        progress_callback(
            stage="index",
            total=total_chunks,
            current=0,
            message="开始构建索引",
            details=f"共 {total_chunks} 个文本块"
        )
        
        # 重置元数据和块ID
        self.metadata = []
        self.chunk_ids = []
        
        # 添加向量到Annoy索引 - 使用连续索引ID
        valid_count = 0
        for i, (chunk, embedding) in enumerate(tqdm(zip(chunks, embeddings), desc="构建索引")):
            if not embedding or len(embedding) != self.dim:
                logger.warning(f"Skipping invalid embedding for chunk {i}")
                continue
                
            try:
                # 确保嵌入是浮点数列表
                embedding = [float(x) for x in embedding]
                
                # 归一化嵌入向量
                embedding_arr = np.array(embedding, dtype=np.float32)
                norm = np.linalg.norm(embedding_arr)
                if norm > 0:
                    embedding_arr = embedding_arr / norm
                else:
                    # 零向量处理 - 跳过无效嵌入
                    logger.warning(f"Zero vector embedding for chunk {i}, skipping")
                    continue
                
                # 使用连续索引ID (valid_count) 而不是文件索引(i)
                self.index.add_item(valid_count, embedding_arr)
                
                # 生成摘要嵌入并添加到摘要索引
                summary_embedding = self._get_summary_embedding(chunk["summary"])
                if summary_embedding and len(summary_embedding) == self.dim:
                    summary_arr = np.array(summary_embedding, dtype=np.float32)
                    summary_norm = np.linalg.norm(summary_arr)
                    if summary_norm > 0:
                        summary_arr = summary_arr / summary_norm
                    self.summary_index.add_item(valid_count, summary_arr)
                else:
                    # 使用主嵌入作为回退
                    self.summary_index.add_item(valid_count, embedding_arr)
                
                self.metadata.append({
                    "text": chunk["text"],
                    "summary": chunk["summary"],
                    "source": chunk["source"]
                })
                self.chunk_ids.append(chunk["chunk_id"])
                valid_count += 1
                
                # 更新进度 - 每10个块更新一次
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
            
        # 构建索引
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
        
        # 保存索引
        self.save_index()
        return True

    def _get_summary_embedding(self, summary: str) -> list:
        """获取摘要的嵌入向量"""
        try:
            # 动态导入避免循环依赖
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
        if self.index is None or self.summary_index is None:
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
        
        # 归一化查询向量
        query_embedding_arr = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_embedding_arr)
        if query_norm > 0:
            query_embedding_arr = query_embedding_arr / query_norm
        else:
            # 零向量处理
            query_embedding_arr = np.zeros(self.dim, dtype=np.float32)
        
        # 1. 使用摘要索引进行初步筛选（获取更多结果）
        try:
            summary_indices, summary_distances = self.summary_index.get_nns_by_vector(
                query_embedding_arr, 
                top_k * 3,  # 获取更多结果用于二次筛选
                include_distances=True,
                search_k=-1
            )
        except Exception as e:
            logger.error(f"Error in summary similarity search: {str(e)}")
            summary_indices, summary_distances = [], []
        
        # 2. 对摘要筛选结果使用主索引进行精排
        results = []
        for idx, angular_dist in zip(summary_indices, summary_distances):
            if idx < len(self.metadata) and idx < len(self.chunk_ids):
                # 计算余弦相似度
                cosine_sim = 1 - (angular_dist ** 2) / 2.0
                cosine_sim = max(-1.0, min(1.0, cosine_sim))
                
                # 获取主索引相似度作为精排分数
                try:
                    item_vector = self.index.get_item_vector(idx)
                    main_cosine_sim = np.dot(query_embedding_arr, item_vector)
                    main_cosine_sim = max(-1.0, min(1.0, main_cosine_sim))
                except:
                    main_cosine_sim = cosine_sim
                
                # 使用主索引相似度作为最终分数
                results.append((
                    main_cosine_sim,  
                    self.chunk_ids[idx], 
                    self.metadata[idx],
                    cosine_sim  # 摘要相似度（用于调试）
                ))
        
        # 按精排分数降序排序并取top_k
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
        return [(score, chunk_id, data) for score, chunk_id, data, _ in sorted_results]
    
    def save_index(self):
        if self.index is None or self.summary_index is None:
            logger.warning("Index is None, cannot save")
            return
            
        self.index.save(str(self.index_path))
        self.summary_index.save(str(self.summary_index_path))
        # 保存元数据和块ID
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": self.metadata,
                "chunk_ids": self.chunk_ids
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved Annoy index with {len(self.metadata)} chunks")