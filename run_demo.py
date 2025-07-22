import subprocess
import webbrowser
import time
import os
import shutil
import sys
from pathlib import Path

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent

# 添加项目根目录到Python路径
sys.path.append(str(BASE_DIR))

def initialize_rag():
    """初始化RAG系统，强制重建向量索引"""
    # 删除现有向量库
    vector_store_dir = BASE_DIR / "data" / "vector_store"
    if vector_store_dir.exists():
        shutil.rmtree(vector_store_dir)
    
    # 导入RAG初始化函数
    from RAG import initialize_rag_system
    
    # 创建RAG系统
    print("Initializing RAG system...")
    rag_retriever = initialize_rag_system(force_rebuild=True)
    print("RAG system initialized successfully")
    return rag_retriever

def run_demo():
    # 初始化RAG系统
    initialize_rag()
    
    # 启动后端
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("Starting backend server...")
    time.sleep(3)  # 等待服务器启动
    
    # 打开前端
    print("Opening chat interface in browser...")
    webbrowser.open('http://localhost:8000')
    
    print("Chat interface opened. Press Ctrl+C to stop.")
    
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        backend_process.terminate()
        print("\nServer stopped")

if __name__ == "__main__":
    run_demo()