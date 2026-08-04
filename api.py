"""
FastAPI Backend — University Services RAG Chatbot
Kết nối RAG Retrieval (Task 9) và Generation (Task 10).

Chạy:
    uvicorn api:app --reload
Hoặc:
    python api.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(
    title="University Services RAG Chatbot API",
    description="API cho chatbot hỏi đáp về dịch vụ và chính sách đại học",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# MODELS
# =============================================================================

class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    conversation_history: Optional[list[dict]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    conversation_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str

# =============================================================================
# IN-MEMORY STORAGE (Đơn giản cho demo, production nên dùng Redis/DB)
# =============================================================================

conversations: dict[str, list[dict]] = {}

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend UI."""
    return FileResponse("index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", message="RAG API is running")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - nhận câu hỏi và trả về câu trả lời từ RAG pipeline.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Import và gọi RAG generation từ Task 10
        from src.task10_generation import generate_with_citation
        
        response = generate_with_citation(
            query=request.query, 
            top_k=request.top_k or 5
        )
        
        answer = response.get("answer", "Chưa thể trả lời.")
        sources = response.get("sources", [])
        
        return ChatResponse(
            answer=answer,
            sources=sources,
        )
        
    except NotImplementedError:
        raise HTTPException(
            status_code=501, 
            detail="Task 10 chưa được implement. Hãy hoàn thành src/task10_generation.py"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi RAG Pipeline: {str(e)}")

@app.post("/chat/with-history", response_model=ChatResponse)
async def chat_with_history(request: ChatRequest):
    """
    Chat endpoint với conversation history cho multi-turn conversation.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    conv_id = "default"  # Simplified, production nên dùng session management
    
    # Lưu vào conversation history
    if conv_id not in conversations:
        conversations[conv_id] = []
    
    # Thêm user message vào history
    conversations[conv_id].append({"role": "user", "content": request.query})
    
    try:
        from src.task10_generation import generate_with_citation
        
        # Xây dựng context từ conversation history
        history_context = ""
        if request.conversation_history:
            for msg in request.conversation_history[-5:]:  # Lấy 5 message gần nhất
                role = "User" if msg["role"] == "user" else "Assistant"
                history_context += f"{role}: {msg['content']}\n"
        
        # Tạo query với context nếu có history
        full_query = request.query
        if history_context:
            full_query = f"Conversation history:\n{history_context}\nCurrent question: {request.query}"
        
        response = generate_with_citation(
            query=full_query, 
            top_k=request.top_k or 5
        )
        
        answer = response.get("answer", "Chưa thể trả lời.")
        sources = response.get("sources", [])
        
        # Thêm assistant response vào history
        conversations[conv_id].append({"role": "assistant", "content": answer})
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conv_id,
        )
        
    except NotImplementedError:
        raise HTTPException(
            status_code=501, 
            detail="Task 10 chưa được implement."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi: {str(e)}")

@app.get("/conversation/{conv_id}")
async def get_conversation(conv_id: str):
    """Lấy lịch sử conversation."""
    if conv_id not in conversations:
        return {"messages": []}
    return {"messages": conversations[conv_id]}

@app.delete("/conversation/{conv_id}")
async def delete_conversation(conv_id: str):
    """Xóa conversation."""
    if conv_id in conversations:
        del conversations[conv_id]
    return {"status": "deleted"}

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG API Server...")
    print("📍 Frontend: http://localhost:8000")
    print("📍 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
