from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.chatbot_router import chatbot_router
from app.routers.book_router import book_router
from app.routers.auth_router import auth_router 
from app.routers.plan_router import plan_router
from app.routers.chat_agent_router import chat_agent_router
from app.routers.festival_router import festival_router

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(chatbot_router, prefix="/api/chatbot")
app.include_router(book_router, prefix="/api/book")
app.include_router(auth_router, prefix="/api/auth")  
app.include_router(plan_router, prefix="/api/plan")
app.include_router(chat_agent_router, prefix="/api/chat")
app.include_router(festival_router, prefix="/api/festival")

@app.get("/")
async def read_root():
    return {"message": "API 서버가 정상적으로 실행 중입니다."}
