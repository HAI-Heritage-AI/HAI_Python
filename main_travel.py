# main_travel.py
from fastapi import FastAPI
from api import travel_j
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI(title="Travel Recommendation API")

app.include_router(travel_j.router, prefix="/travel_j")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_travel:app", host="0.0.0.0", port=8001, reload=True)
