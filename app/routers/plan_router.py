from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from app.plan_agent import plan_travel, calculate_trip_days  # agent에서 함수 가져오기

# APIRouter 인스턴스 생성
plan_router = APIRouter(
    tags=["여행 계획"],
    responses={404: {"description": "Not found"}},
)

# Enum 및 모델 정의
class Gender(str, Enum):
    여성 = "여성"
    남성 = "남성"
    기타 = "기타"

AgeGroup = Enum(
    "AgeGroup",
    {
        "10대": "10대",
        "20대": "20대",
        "30대": "30대",
        "40대": "40대",
        "50대": "50대",
        "60대이상": "60대이상"
    }
)

class Companion(str, Enum):
    혼자 = "혼자"
    연인 = "연인"
    친구 = "친구"
    부모님 = "부모님"
    아이 = "아이"
    기타 = "기타"

class TravelStyle(str, Enum):
    국가유산 = "국가유산"
    휴양 = "휴양"
    액티비티 = "액티비티"
    식도락 = "식도락"
    쇼핑 = "쇼핑"
    SNS감성 = "SNS감성"

class TravelRequest(BaseModel):
    gender: Gender = Field(..., description="여행자의 성별")
    age: AgeGroup = Field(..., description="여행자의 연령대")
    companion: Companion = Field(..., description="동행인 유형")
    destination: str = Field(..., description="여행 목적지", min_length=2)
    style: TravelStyle = Field(..., description="선호하는 여행 스타일")
    startDate: str = Field(..., description="여행 시작 날짜 (YYYY-MM-DD 형식)")
    endDate: str = Field(..., description="여행 종료 날짜 (YYYY-MM-DD 형식)")

    class Config:
        schema_extra = {
            "example": {
                "gender": "female",
                "age": "20대",
                "companion": "친구",
                "destination": "서울",
                "style": "SNS감성",
                "startDate": "2024-11-20",
                "endDate": "2024-11-22"
            }
        }

class TravelResponse(BaseModel):
    status: str
    travel_plan: str
    duration: str

# 여행 계획 생성 엔드포인트
@plan_router.post("/plan", response_model=TravelResponse)
async def generate_travel_plan(request: TravelRequest):
    try:
        # 여행 일수 계산
        nights, days = calculate_trip_days(request.startDate, request.endDate)
        
        user_info = {
            "gender": request.gender,
            "age": request.age.value,
            "companion": request.companion,
            "destination": request.destination,
            "style": request.style,
            "start_date": request.startDate,
            "end_date": request.endDate,
            "duration": f"{nights}박 {days}일"
        }
        
        # 여행 계획 생성
        travel_plan = plan_travel(user_info)

        return TravelResponse(
            status="success",
            travel_plan=travel_plan,
            duration=user_info["duration"]
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
