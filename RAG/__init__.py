# RAG/__init__.py
from .document_loader import DocumentLoader
from .text_splitter import TextSplitter
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .retriever import Retriever
from config import DOCUMENTS_DIR, VECTOR_STORE_DIR
import os
import time
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_rag_system(force_rebuild=False):
    """初始化RAG系统"""
    vector_store = VectorStore(rebuild_mode=force_rebuild)  # 添加重建模式参数
    if not force_rebuild and vector_store.index_path.exists() and vector_store.metadata_path.exists():
        logger.info("Using existing vector store")
        return Retriever()
    
    # 需要构建向量库
    logger.info("Vector store not found or incomplete. Building new vector store...")
    if build_vector_store():
        return Retriever()
    else:
        logger.error("Failed to build vector store")
        return Retriever()

def build_vector_store():
    """手动构建向量存储"""
    logger.info("Building new vector store...")
    
    # 加载文档
    loader = DocumentLoader()
    documents = loader.load_documents()
    if not documents:
        logger.warning("No documents found to build vector store")
        return False
    
    # 分割文档
    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    # 检查是否有文本块
    if not chunks:
        logger.warning("No text chunks created from documents")
        return False
    
    # 生成嵌入
    embedding_model = EmbeddingModel()
    texts = [chunk["text"] for chunk in chunks]
    
    # 分批处理嵌入生成
    batch_size = 32
    embeddings = []
    logger.info("Generating embeddings...")
    start_time = time.time()
    
    # 使用tqdm显示进度条
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embedding_model.embed_texts(batch_texts)
        
        # 检查嵌入是否有效
        valid_embeddings = []
        for emb in batch_embeddings:
            if emb and len(emb) == embedding_model.dim:
                valid_embeddings.append(emb)
            else:
                logger.warning(f"Invalid embedding at batch index {i}")
                valid_embeddings.append([0.0] * embedding_model.dim)  # 使用零向量占位
        
        embeddings.extend(valid_embeddings)
    
    # 检查嵌入数量是否匹配
    if len(embeddings) != len(chunks):
        logger.warning(f"Embeddings count ({len(embeddings)}) doesn't match chunks count ({len(chunks)})")
        # 填充缺失的嵌入
        embeddings.extend([[0.0] * embedding_model.dim] * (len(chunks) - len(embeddings)))
    
    logger.info(f"Embeddings generated in {time.time()-start_time:.2f} seconds")
    
    # 添加到向量存储（使用重建模式）
    vector_store = VectorStore(rebuild_mode=True)  # 启用重建模式
    vector_store.add_chunks(chunks, embeddings)
    
    logger.info("Vector store built successfully")
    return True