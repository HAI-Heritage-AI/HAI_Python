from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.agents import travel_chat_agent  # 전역 TravelChatAgent 인스턴스를 가져옴
import json

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

chat_agent_router = APIRouter(
    tags=["채팅에이전트"]
)

@chat_agent_router.post("/chatagent", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """여행 관련 질문에 답변합니다."""
    try:
        # TravelChatAgent를 통해 답변 생성
        answer = await travel_chat_agent.get_answer(
            question=request.question,
            context=json.dumps(travel_chat_agent.current_travel_plan) if travel_chat_agent.current_travel_plan else None
        )

        return ChatResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
