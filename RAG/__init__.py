from .document_loader import DocumentLoader
from .text_splitter import TextSplitter
from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .retriever import Retriever
from config import DOCUMENTS_DIR, VECTOR_STORE_DIR
import os

def initialize_rag_system(force_rebuild=False):
    """初始化RAG系统"""
    # 检查向量存储是否存在
    vector_store = VectorStore()
    if not force_rebuild and vector_store.index_path.exists():
        print("Using existing vector store")
        return Retriever()
    
    print("Building new vector store...")
    
    # 加载文档
    loader = DocumentLoader()
    documents = loader.load_documents()
    if not documents:
        print("No documents found to build vector store")
        return Retriever()
    
    # 分割文档
    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    # 生成嵌入
    embedding_model = EmbeddingModel()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.embed_texts(texts)
    
    # 添加到向量存储
    vector_store.add_chunks(chunks, embeddings)
    
    print("Vector store built successfully")
    return Retriever()