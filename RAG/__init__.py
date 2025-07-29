
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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_rag_system(force_rebuild=False):
    """
    
    @brief 初始化整个RAG系统，包括文档加载器、分割器、嵌入模型、向量存储和检索器
    
    @param force_rebuild (bool): 是否强制重建向量库，默认为False
    
    @return Retriever: 初始化完成的检索器实例
    """
    vector_store = VectorStore(rebuild_mode=force_rebuild)  
    if not force_rebuild and vector_store.index_path.exists() and vector_store.metadata_path.exists():
        logger.info("Using existing vector store")
        return Retriever()
    
    
    logger.info("Vector store not found or incomplete. Building new vector store...")
    if build_vector_store():
        return Retriever()
    else:
        logger.error("Failed to build vector store")
        return Retriever()

def build_vector_store(progress_callback=None):
    """
    
    @brief 手动触发整个向量存储的构建过程，包括文档加载、文本分割、向量生成和索引构建
    
    @param progress_callback (function, optional): 进度回调函数，用于报告构建进度
    
    @return bool: 构建成功返回True，否则返回False
    """
    logger.info("Building new vector store...")
    
    
    loader = DocumentLoader()
    documents = loader.load_documents()
    
    
    progress_callback = progress_callback or (lambda **kw: None)
    progress_callback(stage="load", total=len(documents), current=0, message="开始加载文档")
    
    if not documents:
        logger.warning("No documents found to build vector store")
        progress_callback(stage="load", message="未找到文档", status="error")
        return False
    
    
    splitter = TextSplitter(progress_callback=progress_callback)
    chunks = splitter.split_documents(documents)
    
    
    if not chunks:
        logger.warning("No text chunks created from documents")
        progress_callback(stage="split", message="未生成文本块", status="error")
        return False
    
    
    embedding_model = EmbeddingModel()
    texts = [chunk["text"] for chunk in chunks]
    
    
    batch_size = 32
    embeddings = []
    logger.info("Generating embeddings...")
    start_time = time.time()
    
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    progress_callback(
        stage="embed",
        total=total_batches,
        current=0,
        message="开始生成嵌入向量",
        details=f"共 {len(texts)} 个文本块，分 {total_batches} 批处理"
    )
    
    
    for batch_idx, i in enumerate(tqdm(range(0, len(texts), batch_size), desc="生成嵌入")):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embedding_model.embed_texts(batch_texts)
        
        
        progress_callback(
            stage="embed",
            current=batch_idx + 1,
            total=total_batches,
            message=f"正在处理第 {batch_idx+1}/{total_batches} 批",
            details=f"文本块 {i+1}-{min(i+batch_size, len(texts))}"
        )
        
        
        valid_embeddings = []
        for emb in batch_embeddings:
            if emb and len(emb) == embedding_model.dim:
                valid_embeddings.append(emb)
            else:
                logger.warning(f"Invalid embedding at batch index {i}")
                valid_embeddings.append([0.0] * embedding_model.dim)  
        
        embeddings.extend(valid_embeddings)
    
    
    if len(embeddings) != len(chunks):
        logger.warning(f"Embeddings count ({len(embeddings)}) doesn't match chunks count ({len(chunks)})")
        
        embeddings.extend([[0.0] * embedding_model.dim] * (len(chunks) - len(embeddings)))
    
    logger.info(f"Embeddings generated in {time.time()-start_time:.2f} seconds")
    
    
    vector_store = VectorStore(rebuild_mode=True)  
    
    
    def index_progress(**kwargs):
        
        progress_callback(**kwargs)
    
    
    success = vector_store.add_chunks(chunks, embeddings, progress_callback=index_progress)
    
    if success:
        logger.info("Vector store built successfully")
        return True
    else:
        logger.error("Failed to build vector store")
        return False