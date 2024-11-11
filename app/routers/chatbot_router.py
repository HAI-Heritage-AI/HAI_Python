# app/routers/chatbot_router.py
from fastapi import APIRouter, Body, Query
from app.Chatbot.chat_with_faiss import process_chat

chatbot_router = APIRouter()

# 기존 함수 (쿼리 파라미터로 받음)
@chatbot_router.post("/chatbot")
async def chatbot_endpoint_query(input_text: str = Query(..., description="User input text")):
    response = process_chat(input_text)
    return {"response": response}

# 새로운 함수 (Request body로 받음)
@chatbot_router.post("/chatbot_body")
async def chatbot_endpoint_body(input_data: dict = Body(..., description="User input text in body")):
    input_text = input_data.get("input_text")
    response = process_chat(input_text)
    return {"response": "수정중 " + response}
