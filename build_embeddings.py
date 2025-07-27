# build_embeddings.py
import logging
from RAG import build_vector_store
import time
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("embedding_builder")

# 修复进度显示函数 - 移除冗余的stage参数
def print_progress(**kwargs):
    """打印进度信息"""
    stage = kwargs.get("stage", "process")
    total = kwargs.get("total", 0)
    current = kwargs.get("current", 0)
    message = kwargs.get("message", "")
    details = kwargs.get("details", "")
    status = kwargs.get("status", "progress")
    
    if stage == "load":
        prefix = "加载文档"
    elif stage == "split":
        prefix = "分割文本"
    elif stage == "embed":
        prefix = "生成嵌入"
    elif stage == "index":
        prefix = "构建索引"
    else:
        prefix = "处理中"
    
    if status == "error":
        symbol = "❌"
    elif status == "completed":
        symbol = "✅"
    else:
        symbol = "🔄"
    
    if total > 0:
        percent = current / total * 100
        progress_bar = f"[{'=' * int(percent/5)}{' ' * (20 - int(percent/5))}] {percent:.1f}%"
        sys.stdout.write(f"\r{symbol} {prefix}: {progress_bar} - {message} {details}")
    else:
        sys.stdout.write(f"\r{symbol} {prefix}: {message} {details}")
    
    sys.stdout.flush()
    
    if status in ["completed", "error"]:
        print()  # 完成时换行

def main():
    logger.info("Starting manual embedding process...")
    start_time = time.time()
    
    # 调用构建函数，传入进度回调
    success = build_vector_store(progress_callback=print_progress)
    
    elapsed = time.time() - start_time
    if success:
        logger.info(f"Embedding process completed successfully in {elapsed:.2f} seconds!")
    else:
        logger.error(f"Embedding process failed in {elapsed:.2f} seconds. Check logs for details.")
        exit(1)

if __name__ == "__main__":
    main()