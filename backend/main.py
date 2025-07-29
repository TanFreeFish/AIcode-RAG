# backend/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
from pathlib import Path
from config import SERVICE_CONFIG

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "backend"))

from ai_service import AIService
from RAG import initialize_rag_system, build_vector_store

app = FastAPI()

# 添加静态文件服务
app.mount("/static", StaticFiles(directory=BASE_DIR / "frontend"), name="static")

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
ai_config = {"model_type": "ollama", "model_name": "qwen:7b"}
ai_service = AIService(ai_config)

class ChatRequest(BaseModel):
    """
    @brief 聊天请求数据模型
    @param message 用户输入的消息
    @param use_rag 是否使用RAG功能
    @param use_rerank 是否使用重排序功能
    """
    message: str
    use_rag: bool = False
    use_rerank: bool = False  # 新增重排序参数

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    @brief 处理聊天请求的端点
    @param request ChatRequest对象，包含用户消息和配置选项
    @return 返回AI生成的回复
    """
    try:
        response = ai_service.generate_response(
            request.message, 
            request.use_rag,
            use_rerank=request.use_rerank
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_config")
async def update_config(new_config: dict):
    """
    @brief 更新AI服务配置的端点
    @param new_config 包含新配置的字典
    @return 配置更新状态
    """
    ai_service.update_config(new_config)
    return {"status": "config updated"}

# 重建索引端点
@app.post("/rebuild_index")
async def rebuild_index():
    """
    @brief 重建RAG索引的端点
    @return 索引重建结果状态
    """
    success = build_vector_store()
    if success:
        ai_service.rag_retriever = initialize_rag_system()
        return {"status": "index rebuilt successfully"}
    else:
        return {"status": "failed to rebuild index", "error": "no documents or chunks found"}

# 上传文档端点
@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """
    @brief 上传文档并自动重建索引的端点
    @param file 上传的文件对象
    @return 文件上传和索引重建结果
    """
    try:
        documents_dir = BASE_DIR / "data" / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)
        file_path = documents_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 自动触发重建索引
        success = build_vector_store()
        if success:
            ai_service.rag_retriever = initialize_rag_system()
            return {"status": "success", "file_path": str(file_path)}
        else:
            return {"status": "file uploaded but failed to rebuild index"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 手动构建嵌入端点
@app.post("/build_embeddings")
async def build_embeddings():
    """
    @brief 手动触发向量嵌入构建过程的端点
    @return 嵌入构建结果状态
    """
    """手动触发向量嵌入过程"""
    success = build_vector_store()
    if success:
        ai_service.rag_retriever = initialize_rag_system()
        return {"status": "embeddings built successfully"}
    else:
        return {"status": "failed to build embeddings", "error": "no documents or chunks found"}
    
# 添加前端服务
@app.get("/")
async def serve_frontend():
    """
    @brief 提供前端页面服务的端点
    @return 前端HTML页面
    """
    return FileResponse(BASE_DIR / "frontend" / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_CONFIG["backend_port"])