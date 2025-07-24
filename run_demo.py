"""
一键启动RAG系统，启动后端服务，打开前端页面
"""
# run_demo.py
import subprocess
import webbrowser
import time
import os
import shutil
import sys
from pathlib import Path
import requests
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent

# 添加项目根目录到Python路径
sys.path.append(str(BASE_DIR))

def initialize_rag():
    """初始化RAG系统"""
    # 检查Ollama服务是否运行
    logger.info("Checking Ollama service...")
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        if response.status_code != 200:
            logger.error("Ollama service not running. Please start Ollama first.")
            return None
    except Exception as e:
        logger.error(f"Ollama service not running: {str(e)}. Please start Ollama first.")
        return None
    
    # 导入RAG初始化函数
    try:
        from RAG import initialize_rag_system
    except ImportError as e:
        logger.error(f"Error importing RAG module: {str(e)}")
        return None
    
    # 创建RAG系统
    logger.info("Initializing RAG system...")
    try:
        rag_retriever = initialize_rag_system(force_rebuild=False)
        logger.info("RAG system initialized successfully")
        return rag_retriever
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return None

def run_demo():
    # 初始化RAG系统
    rag_retriever = initialize_rag()
    if not rag_retriever:
        logger.error("Failed to initialize RAG system. Exiting.")
        return
    
    # 启动后端
    logger.info("Starting backend server...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待服务器启动
    time.sleep(5)
    
    # 检查服务器是否启动
    try:
        response = requests.get("http://localhost:8000", timeout=5)
        if response.status_code == 200:
            logger.info("Backend server started successfully")
        else:
            logger.warning(f"Backend returned status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to connect to backend: {str(e)}")
        backend_process.terminate()
        return
    
    # 打开前端
    logger.info("Opening chat interface in browser...")
    webbrowser.open('http://localhost:8000')
    
    logger.info("Chat interface opened. Press Ctrl+C to stop.")
    
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        backend_process.terminate()
        logger.info("Server stopped")

if __name__ == "__main__":
    run_demo()