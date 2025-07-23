# build_embeddings.py
import logging
from RAG import build_vector_store

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("embedding_builder")

def main():
    logger.info("Starting manual embedding process...")
    
    # 调用RAG模块中的构建函数
    success = build_vector_store()
    
    if success:
        logger.info("Embedding process completed successfully!")
    else:
        logger.error("Embedding process failed. Check logs for details.")
        exit(1)

if __name__ == "__main__":
    main()