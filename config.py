import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DOCUMENTS_DIR = os.path.join(DATA_DIR, 'documents')
VECTOR_STORE_DIR = os.path.join(DATA_DIR, 'vector_store')

# RAG配置
RAG_CONFIG = {
    # 文档加载配置
    "document_loader": {
        "extensions": [".txt", ".pdf", ".docx", ".pptx", ".md"]
    },
    
    # 文本分割配置 - 优化参数
    "text_splitter": {
        "chunk_size": 1500,  # 增大块大小
        "chunk_overlap": 100  # 减小重叠
    },
    
    # 嵌入模型配置 - 使用更快的模型
    "embeddings": {
        "model_type": "ollama",  # ollama 或 huggingface
        "model_name": "all-minilm"  # 更轻量级的模型
    },
    
    # 向量存储配置
    "vector_store": {
        "type": "annoy",  # 确保与实现一致
        "index_name": "document_index"
    },
    
    # 检索器配置
    "retriever": {
        "top_k": 4,
        "score_threshold": 0.6
    }
}