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

def build_vector_store(progress_callback=None):
    """手动构建向量存储"""
    logger.info("Building new vector store...")
    
    # 加载文档
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    # 添加进度回调
    progress_callback = progress_callback or (lambda **kw: None)
    progress_callback(stage="load", total=len(documents), current=0, message="开始加载文档")
    
    if not documents:
        logger.warning("No documents found to build vector store")
        progress_callback(stage="load", message="未找到文档", status="error")
        return False
    
    # 分割文档
    splitter = TextSplitter(progress_callback=progress_callback)
    chunks = splitter.split_documents(documents)
    
    # 检查是否有文本块
    if not chunks:
        logger.warning("No text chunks created from documents")
        progress_callback(stage="split", message="未生成文本块", status="error")
        return False
    
    # 生成嵌入
    embedding_model = EmbeddingModel()
    texts = [chunk["text"] for chunk in chunks]
    
    # 分批处理嵌入生成
    batch_size = 32
    embeddings = []
    logger.info("Generating embeddings...")
    start_time = time.time()
    
    # 添加进度回调
    total_batches = (len(texts) + batch_size - 1) // batch_size
    progress_callback(
        stage="embed",
        total=total_batches,
        current=0,
        message="开始生成嵌入向量",
        details=f"共 {len(texts)} 个文本块，分 {total_batches} 批处理"
    )
    
    # 使用tqdm显示进度条
    for batch_idx, i in enumerate(tqdm(range(0, len(texts), batch_size), desc="生成嵌入")):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embedding_model.embed_texts(batch_texts)
        
        # 更新进度
        progress_callback(
            stage="embed",
            current=batch_idx + 1,
            total=total_batches,
            message=f"正在处理第 {batch_idx+1}/{total_batches} 批",
            details=f"文本块 {i+1}-{min(i+batch_size, len(texts))}"
        )
        
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
    
    # 添加索引构建进度回调 - 修复这里的问题
    def index_progress(**kwargs):
        # 直接传递所有参数，不再添加额外的stage参数
        progress_callback(**kwargs)
    
    # 调用add_chunks并检查返回值
    success = vector_store.add_chunks(chunks, embeddings, progress_callback=index_progress)
    
    if success:
        logger.info("Vector store built successfully")
        return True
    else:
        logger.error("Failed to build vector store")
        return False