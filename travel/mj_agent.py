import os
import requests
from math import radians, sin, cos, sqrt, atan2
from crewai import Agent, Task, Crew
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from typing import Optional
from pydantic import BaseModel
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import datetime

# .env 파일 로드
load_dotenv()

# API 키들을 환경변수에서 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")


def get_user_info():
    user_info = {}
    
    prompts = {
        "gender": ("1. 성별이 어떻게 되세요? (남성/여성): ", ['남성', '여성']),
        "age": ("2. 연령이 어떻게 되세요? (10대/20대/30대/40대/50대/60대이상): ", 
                ['10대', '20대', '30대', '40대', '50대', '60대이상']),
        "companion": ("3. 누구와 함께 가실건가요? (혼자/연인/친구/부모님/자녀/기타): ",
                     ['혼자', '연인', '친구', '부모님', '자녀', '기타']),
        "destination": ("4. 어디로 여행을 가실건가요? (서울/부산/대구/인천/경주): ",
                       ['서울', '부산', '대구', '인천', '경주']),
        "style": ("5. 여행스타일이 어떻게 되세요? (국가유산탐방/힐링/액티비티/식도락/SNS): ",
                  ['국가유산탐방', '힐링', '액티비티', '식도락', 'SNS'])
    }
    
    for key, (prompt, valid_responses) in prompts.items():
        while True:
            response = input(prompt)
            if response in valid_responses:
                user_info[key] = response
                break
            print(f"올바른 옵션을 선택해주세요: {'/'.join(valid_responses)}")
    
    # 날짜 입력 받기
    while True:
        start_date = input("6. 여행 시작 날짜를 입력해주세요 (예: 11월09일): ")
        end_date = input("7. 여행 종료 날짜를 입력해주세요 (예: 11월11일): ")
        
        try:
            start_date_parsed = datetime.strptime(start_date, "%m월%d일")
            end_date_parsed = datetime.strptime(end_date, "%m월%d일")
            
            if start_date_parsed < end_date_parsed:
                user_info["start_date"] = start_date
                user_info["end_date"] = end_date
                duration_days = (end_date_parsed - start_date_parsed).days
                user_info["duration"] = f"{duration_days}박{duration_days+1}일"
                break
            else:
                print("종료 날짜는 시작 날짜보다 이후여야 합니다.")
        except ValueError:
            print("날짜 형식이 올바르지 않습니다. 예: 11월09일")
    
    return user_info


# API 키 직접 설정 부분 제거
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 네이버 API 설정
NAVER_CLIENT_ID = NAVER_CLIENT_ID
NAVER_CLIENT_SECRET = NAVER_CLIENT_SECRET

# Perplexity API 설정 부분을 Tavily로 변경(검색보조용)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Tool의 입력 스키마 정의
class SearchInput(BaseModel):
    query: str

class TravelTimeInput(BaseModel):
    origin: str
    destination: str
    city: str

# 네이버 검색 함수 (메인)
def search_naver(query: str, display: int = 10) -> str:
    blog_url = "https://openapi.naver.com/v1/search/blog"
    place_url = "https://openapi.naver.com/v1/search/local"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    # 장소 검색으로 기본 정보 및 평점 획득
    place_params = {
        "query": query,
        "display": display,
        "sort": "comment"  # 리뷰 수 기준 정렬
    }
    
    # 블로그 검색으로 상세 리뷰 정보 획득
    blog_params = {
        "query": f"{query} 후기",
        "display": display,
        "sort": "sim"
    }
    
    results = []
    
    # 장소 정보 검색
    place_response = requests.get(place_url, headers=headers, params=place_params)
    if place_response.status_code == 200:
        places = place_response.json()['items']
        for place in places:
            # 평점과 리뷰 수 추출
            rating = float(place.get('category1', '0').replace('평점 ', '').replace('/5.0', '') or '0')
            review_count = int(place.get('category2', '0').replace('리뷰 ', '').replace('개', '') or '0')
            
            place_info = {
                'name': place['title'],
                'address': place['address'],
                'rating': rating,
                'review_count': review_count,
                'coordinates': {'x': place['mapx'], 'y': place['mapy']},
                'reviews': []
            }
            
            # 해당 장소의 블로그 리뷰 검색
            blog_response = requests.get(blog_url, headers=headers, params={**blog_params, 'query': f"{place['title']} 후기"})
            if blog_response.status_code == 200:
                reviews = blog_response.json()['items'][:3]
                place_info['reviews'] = reviews
            
            results.append(place_info)
    
    # 결과를 문자열로 포맷팅
    formatted_results = []
    for place in results:
        formatted_results.append(
            f"장소명: {place['name']}\n"
            f"주소: {place['address']}\n"
            f"평점: {place['rating']}\n"
            f"리뷰 수: {place['review_count']}\n"
        )
    
    return "\n".join(formatted_results)

# Perplexity 검색 함수를 Tavily 검색 함수로 변경
def search_tavily(query: str) -> str:
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        max_results=5
    )
    return response.answer

# 동선 최적화 함수 수정
def optimize_route(places_str: str, max_distance: int = 5000) -> str:
    response = tavily_client.search(
        query=f"""
            다음 장소들의 최적 방문 순서를 추천해주세요:
            {places_str}
            
            고려사항:
            1. 이동 거리 최소화 (하루 총 이동시간 2시간 이내)
            2. 높은 평점의 장소 우선
            3. 대중교통 접근성
            4. 각 장소의 운영시간
        """,
        search_depth="advanced",
        include_answer=True
    )
    return response.answer

# 새로운 Tool 추가
tavily_search_tool = Tool(
    name="Tavily Search",
    description="Tavily API를 사용하여 추가 여행 정보를 검색합니다 (보조 검색 도구)",
    func=search_tavily,
    args_schema=SearchInput
)

# 검색 도구 설정
naver_search_tool = Tool(
    name="Naver Search",
    description="네이버 API를 사용하여 여행 정보를 검색합니다 (메인 검색 도구)",
    func=search_naver,
    args_schema=SearchInput
)

# 카카오맵 API 설정
KAKAO_API_KEY = KAKAO_API_KEY

def calculate_travel_time(origin: str, destination: str, city: str) -> dict:
    """카카오맵 API를 사용하여 두 장소 간의 이동 시간을 계산"""
    headers = {
        "Authorization": f"KakaoAK {KAKAO_API_KEY}"
    }
    
    # 장소 -> 좌표 변환
    def get_coordinates(place: str) -> tuple:
        search_url = "https://dapi.kakao.com/v2/local/search/address.json"
        params = {"query": f"{city} {place}"}
        response = requests.get(search_url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            if result["documents"]:
                return (result["documents"][0]["x"], result["documents"][0]["y"])
        return None

    # 경로 탐색 (자동차)
    def get_car_route(origin_coords: tuple, dest_coords: tuple) -> dict:
        route_url = "https://apis-navi.kakaomobility.com/v1/directions"
        params = {
            "origin": f"{origin_coords[0]},{origin_coords[1]}",
            "destination": f"{dest_coords[0]},{dest_coords[1]}",
            "priority": "TIME"
        }
        response = requests.get(route_url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            return {
                "duration": result["routes"][0]["duration"] // 60,  # 분 단위로 변환
                "distance": result["routes"][0]["distance"] // 1000  # km 단위로 변환
            }
        return None

    # 대중교통 경로 탐색
    def get_transit_route(origin_coords: tuple, dest_coords: tuple) -> dict:
        transit_url = "https://apis-navi.kakaomobility.com/v1/directions"
        params = {
            "origin": f"{origin_coords[0]},{origin_coords[1]}",
            "destination": f"{dest_coords[0]},{dest_coords[1]}",
            "priority": "TIME",
            "car_type": "7",  # 대중교통
        }
        response = requests.get(transit_url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            return {
                "duration": result["routes"][0]["duration"] // 60,  # 분 단위로 변환
                "distance": result["routes"][0]["distance"] // 1000  # km 단위로 변환
            }
        return None

    # 도보 시간 계산 (평균 도보 속도 4km/h 가정)
    def calculate_walking_time(distance_km: float) -> int:
        walking_speed = 4  # km/h
        return int((distance_km / walking_speed) * 60)  # 분 단위

    origin_coords = get_coordinates(origin)
    dest_coords = get_coordinates(destination)
    
    if origin_coords and dest_coords:
        car_route = get_car_route(origin_coords, dest_coords)
        transit_route = get_transit_route(origin_coords, dest_coords)
        
        if car_route:
            distance_km = car_route["distance"]
            walking_time = calculate_walking_time(distance_km)
            
            return {
                "도보": f"{walking_time}분",
                "대중교통": f"{transit_route['duration']}분" if transit_route else "경로 없음",
                "자차": f"{car_route['duration']}분",
                "거리": f"{distance_km}km"
            }
    return None

# 계산 에이전트 도구 정의
travel_time_tool = Tool(
    name="Travel Time Calculator",
    func=calculate_travel_time,
    description="두 장소 간의 이동 시간을 계산합니다.",
    args_schema=TravelTimeInput
)


# 2. user_info 먼저 가져오기
user_info = get_user_info()

# 3. LLM 설정
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name="gpt-4-turbo-preview",
    temperature=0.7
)

# 4. 에이전트 정의
personalization_agent = Agent(
    role='여행 맞춤형 추천 전문가',
    goal=f"{user_info['destination']} 맞춤형 여행 계획 수립",
    backstory=f"""당신은 {user_info['destination']} 개인화 여행 전문가입니다.
    사용자의 연령({user_info['age']}), 성별({user_info['gender']}), 
    동행({user_info['companion']}), 선호 스타일({user_info['style']})을 
    고려하여 최적의 여행을 계획합니다.
    모든 응답은 반드시 한국어로만 작성합니다.""",
    tools=[naver_search_tool, tavily_search_tool],
    llm=llm,
    verbose=True,
    system_message="모든 응답은 반드시 한국어로 작성해야 합니다."
)

search_agent = Agent(
    role='여행 정보 검색 전문가',
    goal=f"{user_info['destination']} 여행 정보 수집",
    backstory=f"""당신은 {user_info['destination']} 여행 정보 검색 전문가입니다. 
    네이버 검색을 통해 현지 리뷰, 최신 정보, 실제 경험을 수집하고,
    Tavily를 통해 추가적인 상세 정보를 수집합니다.
    모든 응답은 반드시 한국어로만 작성합니다.""",
    tools=[naver_search_tool, tavily_search_tool],
    llm=llm,
    verbose=True,
    system_message="모든 응답은 반드시 한국어로 작성해야 합니다."
)

calculation_agent = Agent(
    role='계산 에이전트',
    goal='추천지 간의 이동 시간 계산',
    backstory="""당신은 여행지 간의 이동 시간을 계산하는 전문가입니다.
    도보, 대중교통, 자차 등 다양한 이동 수단별 소요 시간을 계산하고,
    최적의 이동 수단을 추천합니다.
    모든 응답은 반드시 한국어로만 작성합니다.""",
    tools=[travel_time_tool],
    llm=llm,
    verbose=True,
    system_message="모든 응답은 반드시 한국어로 작성해야 합니다."
)

planner_agent = Agent(
    role='여행 일정 계획 전문가',
    goal=f"{user_info['destination']}의 {user_info['duration']} 일정 수립",
    backstory=f"""당신은 {user_info['destination']} 여행 전문가입니다.
    사용자가 선택한 {user_info['duration']} 일정에 맞춰
    최적의 동선과 시간 배분을 계획합니다.""",
    llm=llm,
    verbose=True
)


# 5. Task 정의
personalization_task = Task(
    description=f"""
    사용자 정보를 분석하여 {user_info['destination']}에 대한 맞춤형 검색 기준을 설정하세요:
    
    사용자 정보:
    - 성별: {user_info['gender']}
    - 연령: {user_info['age']}
    - 동행: {user_info['companion']}
    - 여행스타일: {user_info['style']}
    
    다음 항목에 대한 검색 기준을 설정해주세요:
    1. 사용자 특성에 맞는 관광지 유형
       - 여행스타일이 '국가유산탐방'이 아닌 경우에도 문화재 1곳 포함
    2. 선호할 만한 맛집 유형
    3. 적합한 액티비티 유형
    4. 숙소 선호 기준
    """,
    expected_output="""
    다음 형식으로 검색 기준을 한국어로 제시해주세요:
    1. 관광지 검색 기준
    2. 맛집 검색 기준
    3. 액티비티 검색 기준
    4. 숙소 검색 기준
    """,
    agent=personalization_agent
)

search_task = Task(
    description=f"""
    personalization_agent가 제시한 기준을 바탕으로 {user_info['destination']}에 대해 다음 정보를 검색하고 정리하세요:
    1. 제시된 기준에 맞는 관광지 검색 (상위 10곳)
    2. 추천된 맛집 유형에 맞는 식당 검색 (반드시 다음 정보 포함)
       - 이름
       - 주소
       - 영업시간
       - 전화번호
       - 추천메뉴
       - 예상비용
    3. 제안된 액티비티 관련 장소 검색
    
    특히 다음 사항을 중점적으로 고려하세요:
    - 평점 3.5 이상, 리뷰 20개 이상인 장소 우선
    - 동선 최적화를 위한 거리 정보
    - 실제 방문자의 생생한 후기
    """,
    expected_output="""
    다음 형식으로 한국어로 응답해주세요:
    1. 맞춤형 관광지 목록 (평점순)
    2. 맞춤형 맛집 목록 (상세 정보 포함)
    3. 추천 액티비티 장소
    """,
    agent=search_agent
)

distance_task = Task(
    description=f"""
    검색 에이전트가 제공한 추천지 목록을 바탕으로, 각 추천지 간의 이동 시간을 계산하세요.
    각 장소 간의 이동에 대해 가장 효율적인 이동 수단을 추천하고, 예상 소요 시간을 계산합니다.
    
    다음 사항을 고려하여 계산해주세요:
    1. 실제 도로 기반 이동 거리
    2. 교통 상황을 반영한 소요 시간
    3. 이동 거리에 따른 적절한 이동 수단 추천
       - 1km 이내: 도보 권장
       - 1~5km: 대중교통 또는 도보 고려
       - 5km 이상: 대중교통 또는 자차 권장
    """,
    expected_output="""
    다음 형식으로 이동 시간을 한국어로 제시해주세요:
    1. 장소A → 장소B:
       - 추천 이동수단: [도보/대중교통/자차]
         (선택 이유: 거리/편의성/효율성 등 구체적인 이유)
       - 예상 소요시간: XX분
    """,
    agent=calculation_agent
)

planning_task = Task(
    description="""
    이전 두 에이전트의 결과물을 통합하여 {destination}에 대한 
    {duration} 일정의 최적 여행 계획을 수립하세요.
    
    최종 결과는 다음 형식으로 출력해주세요:
    일정 정보:
    - 목적지: {destination}
    - 기간: {duration}
    - 여행스타일: {style}
    
    상세 일정:
    1일차:
    - 시간:
    - 장소:
    - 주소:
    - 운영시간:
    - 입장료:
    - 연락처:
    - 소요시간:
    - 이동수단:
    - 이동시간:
    
    추천 맛집:
    - 이름:
    - 주소:
    - 영업시간:
    - 연락처:
    - 추천메뉴:
    - 예상비용:
    
    반드시 다음 사항을 지켜주세요:
    1. 모든 시간은 HH:MM 형식으로 표기
    2. 모든 금액은 숫자와 '원' 단위로 표기
    3. 이동시간은 분 단위로 표기
    4. 모든 필드는 null 값이 없어야 함 (정보가 없는 경우 "정보 없음" 표기)
    5. 이동 수단과 소요 시간은 Travel Time Calculator 결과 반영
    
    1. search_agent의 정보:
    - 평점과 리뷰가 높은 관광지 목록
    - 각 장소의 상세 정보
    - 거리 및 교통 정보
    
    2. personalization_agent의 정보:
    - 사용자 맞춤형 추천 장소들
    - 선정 이유와 특징
    
    다음 사항을 고려하여 최종 일정을 작성하세요:
    1. 높은 평점과 리뷰 수를 가진 장소 우선 배치
    2. 이동 거리 최적화 (인접한 장소들을 묶어서 일정 구성)
    3. 각 장소의 체류 시간과 이동 시간
    4. 식사 시간과 휴식 시간 배분
    5. 대중교통 또는 도보 이동을 고려한 현실적인 일정
    """,
    expected_output="상세 여행 일정표",
    agent=planner_agent
)

# 에이전트 도구 업데이트
planner_agent.tools.append(tavily_search_tool)
planner_agent.tools.append(travel_time_tool)

# Assemble crew
crew = Crew(
    agents=[personalization_agent, search_agent, calculation_agent, planner_agent],
    tasks=[personalization_task, search_task, distance_task, planning_task],
    verbose=True,
    planning=True,
    process_inputs=True,
    share_outputs=True,
    task_dependencies={
        search_task: [personalization_task],
        distance_task: [search_task],
        planning_task: [personalization_task, search_task, distance_task]
    }
)

# Execute tasks
result = crew.kickoff(inputs=user_info)