from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from app.chat_agent import TravelChatAgent
import json

class ChatRequest(BaseModel):
    question: str
    user_info: Optional[Dict[str, Any]] = None

    class Config:
        schema_extra = {
            "example": {
                "question": "여행에 필요한 내용을 입력해주세요.",
                "user_info": {
                    "destination": "경북",
                    "detail_destination": "경주시",
                    "style": "관광",
                    "travel_plan": None
                }
            }
        }

class ChatResponse(BaseModel):
    answer: str

chat_agent_router = APIRouter(
    tags=["채팅에이전트"]
)

chat_agent = TravelChatAgent()

@chat_agent_router.post("/chatagent", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """여행 관련 질문에 답변합니다."""
    try:
        # 사용자 정보가 있고 travel_plan이 없는 경우에만 user_info 설정
        if request.user_info and not request.user_info.get('travel_plan'):
            # 파일에서 로드된 travel_plan 유지
            user_info = request.user_info.copy()
            user_info['travel_plan'] = chat_agent.current_travel_plan
            chat_agent.set_user_info(user_info)

        # 답변 생성
        answer = await chat_agent.get_answer(
            question=request.question,
            context=json.dumps(chat_agent.current_travel_plan) if chat_agent.current_travel_plan else None
        )

        return ChatResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))