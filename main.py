# main.py
from fastapi import FastAPI
from app.routers.chatbot_router import chatbot_router

app = FastAPI()

# /api/chatbot 라우터 등록
app.include_router(chatbot_router, prefix="/api")

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
