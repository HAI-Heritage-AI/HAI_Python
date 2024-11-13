# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.chatbot_router import chatbot_router

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000/"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=[""],
)

/api/chatbot 라우터 등록
app.include_router(chatbot_router, prefix="/api")

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}