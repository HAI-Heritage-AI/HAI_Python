# app/routers/chatbot_router.py
from fastapi import APIRouter, Request
# from app.chatbot.chat_with_faiss import process_chat
from app.chatbot.chat_with_hybrid import process_chat

chatbot_router = APIRouter(
    tags=["챗봇"]
)

# @chatbot_router.post("/chatbot")
# async def chatbot_endpoint(request: Request):
#     data = await request.json()  # JSON 데이터 파싱
#     input_text = data.get("input_text")
#     response = process_chat(input_text)
#     return {"response": response}

@chatbot_router.post("/chatbot")
async def chatbot_endpoint(request: Request):
    data = await request.json()  # JSON 데이터 파싱
    input_text = data.get("input_text")
    response = process_chat(input_text)
    return {"response": response}