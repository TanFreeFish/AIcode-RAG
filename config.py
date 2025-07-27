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
        "extensions": [".txt", ".pdf", ".docx", ".pptx", ".md", ".json", ".csv"]
    },
    
    # 文本分割配置 
    "text_splitter": {
        "chunk_size": 1500, 
        "chunk_overlap": 100  
    },
    
    # 嵌入模型配置 
    "embeddings": {
        "model_type": "ollama",  # ollama 或 huggingface
        "model_name": "all-minilm"  
    },
    
    # 向量存储配置
    "vector_store": {
        "type": "annoy",  
        "index_name": "document_index"
    },
    
    # 检索器配置
    "retriever": {
        "top_k": 20,
        "score_threshold": -1.0
    },
    
    # 新增摘要配置
    "summarizer": {
        "model_name": "qwen:7b",  # 使用Qwen模型生成摘要
        "max_summary_length": 15  # 摘要最大长度（字数）
    }
}