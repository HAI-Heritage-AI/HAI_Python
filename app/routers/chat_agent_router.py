from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.chat_agent import TravelChatAgent
import json

# 채팅 요청을 위한 모델
class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "question": "여행에 필요한 내용을 입력해주세요.",
                "context": "이전 여행 계획 내용..."
            }
        }

class ChatResponse(BaseModel):
    answer: str

# 챗봇 인스턴스 생성
chat_agent = TravelChatAgent()

# APIRouter 생성
chat_agent_router = APIRouter(
    tags=["채팅에이전트"]
)

# 전역 변수로 최근 생성된 여행 계획 저장
latest_travel_plan = None  # 여행 계획이 필요하면 다른 API와 연결 가능

# 채팅 API 엔드포인트
@chat_agent_router.post("/chatagent", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """여행 관련 질문에 답변합니다."""
    try:
        # context가 문자열로 들어올 때 JSON 파싱 처리
        context = request.context
        if context:
            try:
                context = json.loads(context)
                print("Parsed Context:", context)  # 파싱된 context 출력
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="context가 올바른 JSON 형식이 아닙니다.")
        
        # TravelChatAgent를 통해 답변 생성
        answer = await chat_agent.get_answer(
            question=request.question,
            context=context
        )

        return ChatResponse(answer=answer)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
