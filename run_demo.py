import subprocess
import webbrowser
import time
import os

def run_demo():
    # 启动后端
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    backend_process = subprocess.Popen(
        ['uvicorn', 'main:app', '--reload'],
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("Starting backend server...")
    time.sleep(3)  # 等待服务器启动
    
    # 打开前端
    frontend_path = os.path.join(os.path.dirname(__file__), 'frontend', 'index.html')
    webbrowser.open(f'file://{frontend_path}')
    
    print("Chat interface opened in browser. Press Ctrl+C to stop.")
    
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        backend_process.terminate()
        print("\nServer stopped")

if __name__ == "__main__":
    run_demo()