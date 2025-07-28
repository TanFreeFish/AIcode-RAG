import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DOCUMENTS_DIR = os.path.join(DATA_DIR, 'documents')
VECTOR_STORE_DIR = os.path.join(DATA_DIR, 'vector_store')

# 服务配置（全局）
SERVICE_CONFIG = {
    "backend_port": 8000,      # 后端服务端口
    "frontend_port": 3000,     # 前端服务端口
    "ollama_host": "http://localhost:11434",  # Ollama服务地址
    "embedding_timeout": 30,   # 嵌入生成超时时间（秒）
    "rerank_timeout": 120      # 重排序超时时间（秒）
}

# RAG配置（AI组件）
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
        "model_name": "all-minilm", 
        "dim": 384,             # 嵌入维度
        "batch_size": 32         # 批量处理大小
    },
    
    # 向量存储配置
    "vector_store": {
        "type": "annoy",  
        "index_name": "document_index",
        "distance_metric": "angular",  # 距离度量方法
        "build_trees": 10              # Annoy索引树数量
    },
    
    # 检索器配置
    "retriever": {
        "top_k": 20,              # 检索返回的文档数量
        "score_threshold": 0.3,   # 相似度分数阈值
        "enable_rerank": False     # 是否默认启用重排序
    },
    
    # 摘要生成配置
    "summarizer": {
        "model_name": "qwen:7b",   # 摘要生成模型
        "max_summary_length": 15   # 摘要最大长度（字数）
    },
    
    # 修改后的重排序配置 - 二元组格式
    "reranker": {
        "enable": False,           # 是否启用重排序
        "model_name": "qwen:7b",   # 重排序模型
        "top_n_for_rerank": 10,    # 参与重排序的文档数量
        "score_threshold": 0.3,    # 相关性阈值
        "prompt_template": (
                "仅根据问题：'{query}'，对下列摘要按你认为与问题的相关性进行打分，认为不相关的不加入数组。\n"
                "只返回一个JSON数组，格式为[[索引, 分数], [索引, 分数], ...]，对应什么摘要不用输出，按分数降序排列。\n"
                "你是API不能输出任何解释、注释、说明、自然语言也不要对你的输出做任何的解释，只输出JSON数组本身，否则我手上的这只猫就会被我掐死。\n"
                "摘要列表：\n{summaries}"
            )
    }
}