"""
一键启动RAG系统，启动后端服务，打开前端页面
"""
"""run_demo.py
一键启动RAG系统，启动后端服务，打开前端页面
"""
import subprocess
import webbrowser
import time
import os
import shutil
import sys
from pathlib import Path
import requests
import logging
import threading

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent

# 添加项目根目录到Python路径
sys.path.append(str(BASE_DIR))

# 进度显示函数
def demo_progress(stage, total=0, current=0, message="", details="", status="progress"):
    """显示进度信息"""
    if stage == "rag_init":
        prefix = "🧩 RAG初始化"
    elif stage == "server":
        prefix = "🚀 后端服务"
    elif stage == "browser":
        prefix = "🌐 浏览器"
    else:
        prefix = "⚙️  处理中"
    
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

def initialize_rag():
    """初始化RAG系统"""
    # 检查Ollama服务是否运行
    demo_progress(
        stage="rag_init",
        message="检查Ollama服务...",
        status="progress"
    )
    
    try:
        response = requests.get("http://localhost:11434", timeout=5)
        if response.status_code != 200:
            logger.error("Ollama service not running. Please start Ollama first.")
            demo_progress(
                stage="rag_init",
                message="Ollama服务未运行",
                status="error"
            )
            return None
    except Exception as e:
        logger.error(f"Ollama service not running: {str(e)}. Please start Ollama first.")
        demo_progress(
            stage="rag_init",
            message=f"Ollama服务错误: {str(e)}",
            status="error"
        )
        return None
    
    # 导入RAG初始化函数
    try:
        from RAG import initialize_rag_system
    except ImportError as e:
        logger.error(f"Error importing RAG module: {str(e)}")
        demo_progress(
            stage="rag_init",
            message=f"导入RAG模块失败: {str(e)}",
            status="error"
        )
        return None
    
    # 创建RAG系统
    demo_progress(
        stage="rag_init",
        message="初始化系统...",
        status="progress"
    )
    
    try:
        rag_retriever = initialize_rag_system(force_rebuild=False)
        demo_progress(
            stage="rag_init",
            message="RAG系统初始化成功",
            status="completed"
        )
        return rag_retriever
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        demo_progress(
            stage="rag_init",
            message=f"初始化失败: {str(e)}",
            status="error"
        )
        return None

def start_server():
    """启动后端服务器"""
    demo_progress(
        stage="server",
        message="正在启动后端服务...",
        status="progress"
    )
    
    try:
        # 使用Popen启动服务器
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待服务器启动
        time.sleep(3)
        
        # 检查服务器是否启动
        try:
            response = requests.get("http://localhost:8000", timeout=5)
            if response.status_code == 200:
                demo_progress(
                    stage="server",
                    message="后端服务启动成功",
                    status="completed"
                )
                return process
            else:
                demo_progress(
                    stage="server",
                    message=f"后端返回状态码: {response.status_code}",
                    status="error"
                )
                return None
        except Exception as e:
            demo_progress(
                stage="server",
                message=f"连接后端失败: {str(e)}",
                status="error"
            )
            return None
    except Exception as e:
        demo_progress(
            stage="server",
            message=f"启动服务器失败: {str(e)}",
            status="error"
        )
        return None

def run_demo():
    # 初始化RAG系统
    rag_retriever = initialize_rag()
    if not rag_retriever:
        logger.error("Failed to initialize RAG system. Exiting.")
        return
    
    # 启动后端
    server_process = start_server()
    if not server_process:
        logger.error("Failed to start backend server. Exiting.")
        return
    
    # 打开前端
    demo_progress(
        stage="browser",
        message="正在打开聊天界面...",
        status="progress"
    )
    time.sleep(1)  # 确保服务器完全启动
    
    try:
        webbrowser.open('http://localhost:8000')
        demo_progress(
            stage="browser",
            message="聊天界面已打开",
            status="completed"
        )
    except Exception as e:
        demo_progress(
            stage="browser",
            message=f"打开浏览器失败: {str(e)}",
            status="error"
        )
    
    logger.info("Chat interface opened. Press Ctrl+C to stop.")
    
    try:
        # 打印服务器日志
        def log_stream(stream, prefix):
            for line in stream:
                if line:  # 确保行不为空
                    logger.info(f"{prefix}: {line.strip()}")
        
        # 启动线程捕获stdout和stderr
        stdout_thread = threading.Thread(
            target=log_stream, 
            args=(server_process.stdout, "SERVER"),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=log_stream, 
            args=(server_process.stderr, "ERROR"),
            daemon=True
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # 等待服务器进程结束
        server_process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        server_process.terminate()
        logger.info("Server stopped")

if __name__ == "__main__":
    run_demo()