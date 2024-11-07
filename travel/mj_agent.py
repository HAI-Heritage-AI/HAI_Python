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
                  ['국가유산탐방', '힐링', '액티비티', '식도락', 'SNS']),
        "duration": ("6. 여행 일정을 선택해주세요 (1박2일/2박3일/3박4일): ",
                    ['1박2일', '2박3일', '3박4일'])
    }
    
    for key, (prompt, valid_responses) in prompts.items():
        while True:
            response = input(prompt)
            if response in valid_responses:
                user_info[key] = response
                break
            print(f"올바른 옵션을 선택해주세요: {'/'.join(valid_responses)}")
    
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
    
    # 평점과 리뷰 수를 고려한 점수 계산
    for result in results:
        result['score'] = (result['rating'] * 0.7) + (min(result['review_count'], 100) / 100 * 0.3)
    
    # 점수 기준 정렬
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 결과를 문자열로 포맷팅
    formatted_results = []
    for place in results:
        reviews_text = "\n".join([f"- {review['description']}" for review in place['reviews']])
        formatted_results.append(
            f"장소명: {place['name']}\n"
            f"주소: {place['address']}\n"
            f"평점: {place['rating']}\n"
            f"리뷰 수: {place['review_count']}\n"
            f"종합 점수: {place['score']:.2f}\n"
            f"리뷰:\n{reviews_text}\n"
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

    # 경로 탐색
    origin_coords = get_coordinates(origin)
    dest_coords = get_coordinates(destination)
    
    if origin_coords and dest_coords:
        route_url = "https://apis-navi.kakaomobility.com/v1/directions"
        params = {
            "origin": f"{origin_coords[0]},{origin_coords[1]}",
            "destination": f"{dest_coords[0]},{dest_coords[1]}",
            "priority": "TIME"  # 최단 시간 경로
        }
        
        response = requests.get(route_url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            return {
                "duration": result["routes"][0]["duration"],  # 초 단위
                "distance": result["routes"][0]["distance"],  # 미터 단위
                "transport": result["routes"][0]["transport_type"]
            }
    return None

# 새로운 Tool 추가
travel_time_tool = Tool(
    name="Travel Time Calculator",
    description="두 장소 간의 이동 시간을 계산합니다",
    func=calculate_travel_time,
    args_schema=SearchInput
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
planner_agent = Agent(
    role='여행 일정 계획 전문가',
    goal=f"{user_info['destination']}의 {user_info['duration']} 일정 수립",
    backstory=f"""당신은 {user_info['destination']} 여행 전문가입니다.
    사용자가 선택한 {user_info['duration']} 일정에 맞춰
    최적의 동선과 시간 배분을 계획합니다.""",
    llm=llm,
    verbose=True
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

# 개인 맞춤형 에이전트 정의
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

planner_agent = Agent(
    role='여행 일정 계획 전문가',
    goal=f"{user_info['destination']}의 {user_info['duration']} 일정 수립",
    backstory=f"""당신은 {user_info['destination']} 여행 전문가입니다.
    사용자가 선택한 {user_info['duration']} 일정에 맞춰
    최적의 동선과 시간 배분을 계획합니다.
    모든 응답은 반드시 한국어로만 작성합니다.""",
    llm=llm,
    verbose=True,
    system_message="모든 응답은 반드시 한국어로 작성해야 합니다."
)

# 5. Task 정의
search_task = Task(
    description="""
    {destination}에 대해 다음 정보를 검색하고 정리하세요:
    1. 평점과 리뷰 수가 높은 관광지 추천 (상위 10곳)
    2. 각 장소의 실제 방문자 리뷰 요약
    3. 주요 관광지간 거리 및 이동 시간
    4. 각 장소의 운영시간과 입장료
    5. 대중교통 접근성
    
    특히 다음 사항을 중점적으로 고려하세요:
    - 평점 3.5 이상, 리뷰 20개 이상인 장소 우선
    - 동선 최적화를 위한 거리 정보
    - 실제 방문자의 생생한 후기
    """,
    expected_output="""
    다음 형식으로 한국어로 응답해주세요:
    1. 추천 관광지 목록 (평점순)
    2. 각 장소별 상세 정보
    3. 방문자 리뷰 요약
    4. 교통 및 접근성 정보
    """,
    agent=search_agent
)

personalization_task = Task(
    description="""
    search_agent가 제공한 정보를 바탕으로, 다음 사용자 정보에 맞는 맞춤형 추천을 제공하세요:
    - 성별: {gender}
    - 연령: {age}
    - 동행: {companion}
    - 여행스타일: {style}
    
    다음 항목을 포함하여 추천해주세요:
    1. 사용자 특성에 맞는 관광지 3-5곳 (search_agent의 결과에서 선별)
       - 만약 여행스타일이 '국가유산탐방'이 아닌 경우, 반드시 1곳은 국가유산을 포함할 것
       - 여행스타일이 '국가유산탐방'인 경우, 문화재와 역사적 장소 위주로 추천
    2. 취향에 맞는 맛집 3-5곳
    3. 동행자와 함께하기 좋은 액티비티
    4. 선호 스타일에 맞는 숙소 추천
    """,
    expected_output="""
    다음 형식으로 한국어로 응답해주세요:
    1. 맞춤형 관광지 추천
    2. 맛집 추천 목록
    3. 추천 액티비티
    4. 숙소 추천
    """,
    agent=personalization_agent
)

planning_task = Task(
    description=f"""
    이전 두 에이전트의 결과물을 통합하여 {user_info['destination']}에 대한 
    {user_info['duration']} 일정의 최적 여행 계획을 수립하세요:
    
    반드시 다음 사항을 지켜주세요:
    - 목적지는 {user_info['destination']}만 포함할 것
    - 일정은 정확히 {user_info['duration']}에 맞출 것
    
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
    
    만약 인기 장소들의 동선이 비효율적이라면:
    - 차순위 인기 장소들을 고려하여 더 효율적인 동선 구성
    - 이동 시간 대비 장소의 매력도 평가
    - 하루 총 이동 시간이 1시간을 초과하지 않을 것
    - 각 장소 간 이동 시간을 Travel Time Calculator로 확인할 것
    - 이동 시간이 30분을 초과하는 경우 대안 장소를 고려할 것
    """,
    expected_output="상세 일정표 (각 이동 구간의 소요 시간 포함)",
    agent=planner_agent
)

# 에이전트 도구 업데이트
planner_agent.tools.append(tavily_search_tool)
planner_agent.tools.append(travel_time_tool)

# Assemble crew
crew = Crew(
    agents=[search_agent, personalization_agent, planner_agent],
    tasks=[search_task, personalization_task, planning_task],
    verbose=True,
    planning=True,
    process_inputs=True,  # 입력 정보 공유
    share_outputs=True,   # 출력 정보 공유
    task_dependencies={
        personalization_task: [search_task],  # personalization_task는 search_task의 결과에 의존
        planning_task: [search_task, personalization_task]  # planning_task는 두 task의 결과에 의존
    }
)

# Execute tasks
result = crew.kickoff(inputs=user_info)
