from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from app.plan_agent import plan_travel, calculate_trip_days  # 외부 함수 가져오기
from app.agents import travel_chat_agent
import httpx  # API 호출을 위한 httpx 사용
from datetime import datetime
import logging
import json
import os

logging.basicConfig(level=logging.DEBUG)

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
        "60대이상": "60대이상",
    },
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
    detail_destination: str = Field(..., description="여행 세부 목적지", min_length=2)
    style: TravelStyle = Field(..., description="선호하는 여행 스타일")
    startDate: datetime = Field(..., description="여행 시작 날짜 (YYYY-MM-DD 형식)")
    endDate: datetime = Field(..., description="여행 종료 날짜 (YYYY-MM-DD 형식)")

    class Config:
        schema_extra = {
            "example": {
                "gender": "여성",
                "age": "20대",
                "companion": "친구",
                "destination": "서울",
                "detail_destination": "강남",
                "style": "SNS감성",
                "startDate": "2024-11-20",
                "endDate": "2024-11-22",
            }
        }

class TravelResponse(BaseModel):
    status: str
    travel_plan: dict
    duration: str
    festivals: list[dict] = []

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
            "detail_destination": request.detail_destination,
            "style": request.style,
            "start_date": request.startDate,
            "end_date": request.endDate,
            "duration": f"{nights}박 {days}일",
        }

        # 여행 계획 생성
        travel_plan = plan_travel(user_info)

        # travel_plan이 JSON 문자열인 경우 파싱하여 dict로 변환
        if isinstance(travel_plan, str):
            try:
                travel_plan = json.loads(travel_plan)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail="여행 계획 데이터를 파싱하는 데 실패했습니다.")

        # travel_plan이 dict 형태인지 검증
        if not isinstance(travel_plan, dict):
            raise HTTPException(status_code=500, detail="여행 계획 데이터가 올바른 형식이 아닙니다.")

        # "result" 키가 포함된 경우 해당 데이터를 분리
        if "result" in travel_plan:
            travel_plan = travel_plan["result"]

        # 각 Day의 값이 리스트(배열)인지 확인하고, 아니면 빈 리스트로 설정
        for day, events in travel_plan.items():
            if not isinstance(events, list):
                logging.warning(f"Day '{day}'의 데이터가 배열이 아닙니다. 빈 배열로 설정합니다.")
                travel_plan[day] = []

        # 여행 시작일과 종료일을 문자열로 변환
        start_date_str = request.startDate.strftime("%Y-%m-%d")
        end_date_str = request.endDate.strftime("%Y-%m-%d")

        # TravelChatAgent에 최신 여행 계획 설정
        travel_chat_agent.set_user_info(user_info)
        
        # 축제 데이터 가져오기
        async with httpx.AsyncClient() as client:
            festival_response = await client.get(
                "http://localhost:8000/api/festival/",
                params={"destination": request.destination, "start_date": start_date_str},
            )
            if festival_response.status_code != 200:
                raise HTTPException(
                    status_code=festival_response.status_code,
                    detail="축제 데이터를 가져오는 데 실패했습니다.",
                )
            festival_data = festival_response.json()

        # 여행 계획 + 축제 데이터 병합
        response_data = {
            "status": "success",
            "travel_plan": travel_plan,
            "duration": user_info["duration"],
            "festivals": festival_data,
        }

        # data 디렉토리 생성 및 plan.json 파일 저장
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','travel', 'data')
        os.makedirs(data_dir, exist_ok=True)
        output_file = os.path.join(data_dir, 'plan.json')

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)

        print(f"결과가 {output_file}에 저장되었습니다.")

        # 여행 계획 + 축제 데이터 반환
        return TravelResponse(**response_data)

    except ValueError as ve:
        logging.error(f"Unexpected error: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"잘못된 요청입니다: {str(ve)}")
    except TypeError as te:
        logging.error(f"Unexpected error: {str(te)}")
        raise HTTPException(status_code=400, detail=f"타입 오류가 발생했습니다: {str(te)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"서버 오류가 발생했습니다: {str(e)}")
