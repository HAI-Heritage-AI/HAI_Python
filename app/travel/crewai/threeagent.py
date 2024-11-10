
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from crewai import Agent, Task, Crew
from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    SerperDevTool,
    WebsiteSearchTool
)

app = FastAPI()

class TravelRequest(BaseModel):
    gender: str
    age: str
    companion: str
    destination: str
    style: str
    start_date: str
    end_date: str

@app.post("/create_travel_plan")
async def create_travel_plan(travel_request: TravelRequest):
    try:
        # API 키 설정
        serper_api_key = os.getenv("SERPER_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")


        # Tools 초기화
        search_tool = SerperDevTool()
        web_rag_tool = WebsiteSearchTool()

        # 날짜 처리
        start_date = travel_request.start_date
        end_date = travel_request.end_date
        
        # datetime 객체 생성
        current_year = datetime.now().year
        start_month, start_day = map(int, start_date.split('/'))
        end_month, end_day = map(int, end_date.split('/'))
        
        start = datetime(current_year, start_month, start_day)
        end = datetime(current_year, end_month, end_day)
        
        if end < start:
            end = datetime(current_year + 1, end_month, end_day)
        
        nights = (end - start).days
        days = nights + 1
        duration = f"{nights}박 {days}일"

        # user_info 딕셔너리 생성
        user_info = {
            "gender": travel_request.gender,
            "age": travel_request.age,
            "companion": travel_request.companion,
            "destination": travel_request.destination,
            "style": travel_request.style,
            "start_date": start.strftime("%m/%d"),
            "end_date": end.strftime("%m/%d"),
            "duration": duration
        }

        # Create agents
        general_researcher = Agent(
            role='일반 여행 조사 에이전트',
            goal='선택한 {destination}의 전반적인 정보와 주소 제공',
            backstory='여행지의 기본 정보와 주요 관광지를 조사하는 전문가입니다.',
            tools=[search_tool, web_rag_tool],
            verbose=True
        )

        personal_researcher = Agent(
            role='맞춤형 여행 조사 에이전트',
            goal='{gender}, {age}의 {companion}과 함께하는 {style} 스타일의 여행 정보 제공',
            backstory="""사용자의 특성과 선호도를 고려하여 맞춤형 여행 정보를 조사하는 전문가입니다.
            특히 계절에 맞지 않는 부적절한 활동은 제외하고 추천합니다.
            예를 들어:
            - 여름: 스키장, 눈썰매장 제외
            - 겨울: 워터파크, 해수욕장 제외
            - 봄/가을: 계절에 맞는 축제와 야외활동 위주로 추천""",
            tools=[search_tool, web_rag_tool],
            verbose=True
        )

        itinerary_writer = Agent(
            role='여행 일정 작성자',
            goal='입력된 날짜({start_date}~{end_date})에 맞춘 {duration} 일정 작성',
            backstory="""수집된 정보를 바탕으로 최적화된 여행 일정을 작성하는 전문가입니다.
            모든 이동 시간은 대중교통/택시 기준 1시간 이내로 제한하여 효율적인 동선을 계획합니다.""",
            verbose=True
        )

        # Define tasks
        general_research_task = Task(
            description="""
            {destination}에 대해 다음 정보를 조사하세요:
            1. 주요 관광지 5곳과 주소
            2. 운영 시간
            3. 입장료
            4. 교통 정보
            5. 편의시설
            모든 정보는 반드시 한국어로 작성하세요.
            """,
            expected_output="한국어로 작성된 관광지의 기본 정보와 주소가 포함된 상세 보고서",
            agent=general_researcher
        )

        personal_research_task = Task(
            description="""
            다음 사용자 특성에 맞는 추천 정보를 조사하세요:
            - 성별: {gender}
            - 연령: {age}
            - 동행: {companion}
            - 여행스타일: {style}
            - 여행 시작일: {start_date}
            
            1. 맞춤형 관광지 추천 (서로 30분 이내 거리)
            2. 식당 추천 (현재 계절 메뉴 고려)
            3. 쇼핑 장소
            4. 계절에 맞는 특별 활동이나 축제
            
            모든 정보는 반드시 한국어로 작성하세요.
            """,
            expected_output="한국어로 작성된 계절을 고려한 맞춤형 추천 정보 보고서",
            agent=personal_researcher
        )

        write_task = Task(
            description="""
            앞선 두 에이전트가 추천한 장소들을 바탕으로 {duration} 일정을 계획해주세요.
            
            우선 추천받은 모든 장소들의 정확한 정보를 조사하세요:
            1. 정확한 도로명 주소와 지번 주소 (네이버/카카오맵 검색 기준)
            2. 영업시간
            3. 전화번호
            4. 가장 가까운 대중교통 정보
            
            그리고 다음 내용으로 여행 계획을 작성하세요:
            - 대상: {gender}, {age}, {companion}과 동행
            - 여행스타일: {style}
            - 기간: {start_date}부터 {end_date}까지 {duration}
            
            일정 작성 시 주의사항:
            1. 모든 장소는 검증된 정확한 주소 사용
            2. 이동시간은 실제 대중교통 기준으로 계산 (1시간 이내로 제한)
            3. 각 장소 간 이동경로 상세히 기록
            4. 식사 장소는 인근 맛집으로 선정
            5. 동선이 효율적이도록 인접 장소끼리 묶어서 계획
            
            최종 일정에는 다음이 포함되어야 합니다:
            1. 날짜/시간별 세부 일정
            2. 각 장소의 정확한 주소
            3. 상세 이동 방법과 소요시간
            4. 식사 정보와 예상 비용
            5. 전체 예상 비용

            모든 내용은 한국어로 작성하며, 금액은 원화로 표시해주세요.
            """,
            expected_output="검증된 주소와 상세 동선이 포함된 여행 일정",
            agent=itinerary_writer,
            output_file='itinerary/personalized_itinerary.md'
        )

        # Assemble and execute crew
        crew = Crew(
            agents=[general_researcher, personal_researcher, itinerary_writer],
            tasks=[general_research_task, personal_research_task, write_task],
            verbose=True,
            planning=True,
        )

        result = crew.kickoff(inputs=user_info)
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)