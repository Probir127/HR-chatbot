# api_server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import backend
import os, json

app = FastAPI(title="HR Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[List[Message]] = []

@app.get("/")
def root():
    return {"status":"HR Chatbot API Running"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # keep compatibility with previous call signature (ask_hr_bot)
        # pass chat_history if needed later
        reply = backend.ask_hr_bot(user_input=request.message, chat_history=[{"role":m.role,"content":m.content} for m in request.chat_history])
        return {"response": reply}
    except Exception as e:
        print("❌ Chat Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status":"ok"}
