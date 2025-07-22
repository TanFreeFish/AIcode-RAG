from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_service import AIService, RAGService

app = FastAPI()

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
rag_service = RAGService({})

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = False

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        rag_context = rag_service.retrieve_context(request.message) if request.use_rag else None
        response = ai_service.generate_response(request.message, rag_context)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_config")
async def update_config(new_config: dict):
    ai_service.update_config(new_config)
    return {"status": "config updated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)