# app/routers/chatbot_router.py
from fastapi import APIRouter
from app.chatbot.chat import process_chat

chatbot_router = APIRouter()

@chatbot_router.post("/chatbot")
async def chatbot_endpoint(input_text: str):
    response = process_chat(input_text)
    return {"response": response}
