"""run_demo.py
一键启动RAG系统，启动后端服务，打开前端页面
"""
import subprocess
import webbrowser
import time
import sys
from pathlib import Path
import requests
import logging
import threading
from config import SERVICE_CONFIG


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).resolve().parent


sys.path.append(str(BASE_DIR))


def demo_progress(stage, total=0, current=0, message="", details="", status="progress"):
    """
    @brief 显示进度信息
    
    @param stage 进度阶段
    @param total 总数
    @param current 当前进度
    @param message 消息内容
    @param details 详细信息
    @param status 状态
    """
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
        print()  

def initialize_rag():
    """
    @brief 初始化RAG系统
    
    @return 初始化后的RAG检索器，失败时返回None
    """
    
    demo_progress(
        stage="rag_init",
        message="检查Ollama服务...",
        status="progress"
    )
    
    try:
        response = requests.get(f"{SERVICE_CONFIG['ollama_host']}", timeout=5)
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
    """
    @brief 启动后端服务器
    
    @return 服务器进程对象，失败时返回None
    """
    demo_progress(
        stage="server",
        message="正在启动后端服务...",
        status="progress"
    )
    
    try:
        
        process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", "backend.main:app",
                "--host", "0.0.0.0", "--port", str(SERVICE_CONFIG["backend_port"])
            ],
            cwd=BASE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        
        time.sleep(3)
        
        
        try:
            response = requests.get(
                f"http://localhost:{SERVICE_CONFIG['backend_port']}", 
                timeout=5
            )
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
    """
    @brief 运行演示程序，包括初始化RAG系统、启动后端服务和打开前端页面
    """
    
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
    time.sleep(1)  
    
    try:
        webbrowser.open(f'http://localhost:{SERVICE_CONFIG["backend_port"]}')
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
        
        def log_stream(stream, prefix):
            """
            @brief 日志流处理函数
            
            @param stream 日志流
            @param prefix 前缀标识
            """
            for line in stream:
                if line:  
                    logger.info(f"{prefix}: {line.strip()}")
        
        
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
        
        
        server_process.wait()
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        server_process.terminate()
        logger.info("Server stopped")

if __name__ == "__main__":
    run_demo()