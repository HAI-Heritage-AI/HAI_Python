from crewai import Agent, Task, Crew
from crewai_tools import BaseTool, SerperDevTool, CSVSearchTool
from typing import Optional
import requests
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import unicodedata
import re

# 환경변수 로딩 및 검사
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 기준의 절대 경로
TRAVEL_DATA_DIR = os.path.join(BASE_DIR,  'travel','data')  # travel/data 디렉토리 경로

# API 키 확인
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY not found in environment variables")
if not os.getenv("NAVER_CLIENT_ID") or not os.getenv("NAVER_CLIENT_SECRET"):
    raise ValueError("NAVER API credentials not found in environment variables")
if not os.getenv("KAKAO_REST_API_KEY"):
    raise ValueError("KAKAO_REST_API_KEY not found in environment variables")

def get_csv_file_paths(destination: str) -> dict:
    """
    주어진 목적지(destination)에 해당하는 여행지와 맛집 CSV 파일 경로를 반환합니다.
    """
    base_paths = {
        'travel': os.path.join(TRAVEL_DATA_DIR, 'travel'),
        'food': os.path.join(TRAVEL_DATA_DIR, 'food'),
    }
    
    result = {'travel': None, 'food': None}
    for category, base_path in base_paths.items():
        if not os.path.exists(base_path):
            print(f"Error: '{base_path}' 경로가 존재하지 않습니다.")
            continue

        print(f"'{base_path}' 경로에서 파일을 검색 중...")
        normalized_destination = unicodedata.normalize('NFC', destination)

        for file_name in os.listdir(base_path):
            normalized_file_name = unicodedata.normalize('NFC', file_name)
            if normalized_destination in normalized_file_name and normalized_file_name.endswith('.csv'):
                print(f"{destination}에 해당하는 {category} 파일 찾음: {file_name}")
                result[category] = os.path.join(base_path, file_name)
                break

        if result[category] is None:
            print(f"{destination}에 해당하는 {category} CSV 파일을 찾을 수 없습니다.")

    return result

import pandas as pd

def convert_csv_to_utf8(original_csv_path: str, temp_csv_path: str) -> None:
    """
    CSV 파일을 UTF-8로 변환하여 임시 파일로 저장합니다.
    """
    try:
        # 파일을 'utf-8' 인코딩으로 시도해서 읽기
        df = pd.read_csv(original_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 만약 'utf-8'로 읽기 실패하면 'euc-kr'로 시도
        try:
            df = pd.read_csv(original_csv_path, encoding='euc-kr')
        except UnicodeDecodeError:
            # 'euc-kr'도 실패하면 'latin1'로 시도
            df = pd.read_csv(original_csv_path, encoding='latin1')

    # UTF-8로 저장
    df.to_csv(temp_csv_path, encoding='utf-8', index=False)
    print(f"{original_csv_path} 파일을 UTF-8로 변환하여 {temp_csv_path}에 저장했습니다.")

def calculate_trip_days(start_date, end_date):
    """
    여행 일수를 계산하는 함수
    YYYY-MM-DD 형식의 날짜를 처리하며, 연도와 월이 바뀌는 경우도 처리
    """
    try:
        # 만약 start_date와 end_date가 문자열이면 datetime 객체로 변환
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 시작일과 종료일의 차이 계산
        date_diff = end_date - start_date
        nights = date_diff.days
        days = nights + 1

        # 유효성 검사
        if days <= 0:
            raise ValueError("종료일이 시작일보다 빠릅니다.")
        if days > 365:
            raise ValueError("여행 기간이 1년을 초과할 수 없습니다.")
            
        # 날짜 정보 디버깅
        print(f"여행 정보:")
        print(f"시작일: {start_date.strftime('%Y년 %m월 %d일')}")
        print(f"종료일: {end_date.strftime('%Y년 %m월 %d일')}")
        print(f"총 {nights}박 {days}일")
        
        # 연도나 월이 바뀌는지 확인
        if start_date.year != end_date.year:
            print(f"주의: 연도가 바뀌는 여행입니다 ({start_date.year}년 → {end_date.year}년)")
        elif start_date.month != end_date.month:
            print(f"주의: 월이 바뀌는 여행입니다 ({start_date.month}월 → {end_date.month}월)")
        
        return (nights, days)
        
    except ValueError as e:
        print(f"날짜 오류: {e}")
        print("YYYY-MM-DD 형식으로 입력해주세요 (예: 2024-11-20)")
        return (0, 0)
    except Exception as e:
        print(f"예상치 못한 오류 발생: {e}")
        return (0, 0)
    

class KakaoLocalSearchTool(BaseTool):
    """카카오 로컬 API를 이용한 좌표 검색 도구"""
    name: str = "Kakao Local Search"
    description: str = "카카오 로컬 API로 주소를 검색하여 좌표를 반환합니다."
    api_key: str = ""  # 필드 선언 추가
    headers: dict = {}  # 필드 선언 추가

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("KAKAO_REST_API_KEY")
        if not self.api_key:
            raise ValueError("KAKAO_REST_API_KEY not found in environment variables")
        self.headers = {
            "Authorization": f"KakaoAK {self.api_key}"
        }

    def _run(self, address: str) -> str:
        """BaseTool 요구사항을 충족하기 위한 메소드"""
        result = self.get_coordinates(address)
        return json.dumps(result, ensure_ascii=False) if result else json.dumps({"error": "주소를 찾을 수 없습니다."})


    def get_coordinates(self, address: str) -> dict:
        """주소를 검색하여 좌표를 반환합니다."""
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        params = {"query": address}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            result = response.json()
            
            if result.get('documents'):
                document = result['documents'][0]
                return {
                    "address_name": document.get('address_name', ''),
                    "x": document.get('x'),  # 경도
                    "y": document.get('y')   # 위도
                }
            return None
            
        except Exception as e:
            print(f"카카오 API 호출 중 오류 발생: {e}")
            return None

class NaverLocalSearchTool(BaseTool):
    """네이버 지역 검색과 카카오 좌표 변환 통합 도구"""
    name: str = "Naver Local Search"
    description: str = "네이버 지역 검색으로 장소를 검색하고 카카오 API로 좌표를 조회합니다."
    client_id: str = ""  # 필드 선언 추가
    client_secret: str = ""  # 필드 선언 추가
    headers: dict = {}  # 필드 선언 추가
    kakao_tool: KakaoLocalSearchTool = None  # 필드 추가


    def __init__(self):
        super().__init__()
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        self.kakao_tool = KakaoLocalSearchTool()


    def _run(self, query: str) -> str:
        url = "https://openapi.naver.com/v1/search/local"
        params = {
            "query": query,
            "display": 10,
            "sort": "random"
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            items = response.json().get('items', [])
            
            results = []
            for item in items:
                place_info = {
                    "name": item['title'].replace('<b>', '').replace('</b>', ''),
                    "address": item['address'],
                    "category": item.get('category', '정보 없음'),
                    "roadAddress": item.get('roadAddress', '정보 없음'),
                    "telephone": item.get('telephone', '정보 없음')
                }
                
                # 카카오 API로 좌표 조회
                coordinates = self.kakao_tool.get_coordinates(item['address'])
                if coordinates:
                    place_info.update({
                        "address_name": coordinates['address_name'],
                        "x": coordinates['x'],
                        "y": coordinates['y']
                    })
                
                results.append(place_info)
            
            return json.dumps({
                "places": results
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

def create_travel_agents(llm, user_info):
    # Serper Tool 초기화
    search_tool = SerperDevTool()
    
    # 네이버 로컬 검색 도구 초기화 (카카오 좌표 변환 포함)
    local_tool = NaverLocalSearchTool()

    # 목적지에 해당하는 CSV 파일 경로 가져오기
    destination = user_info["destination"]
    style = user_info["style"]

    csv_paths = get_csv_file_paths(destination)

    if not csv_paths['travel'] and not csv_paths['food']:
        print(f"{destination}에 해당하는 CSV 파일을 찾을 수 없습니다.")
        return None, None, None, None


    # DataFrame으로 직접 로드하여 메모리에서 처리
    def load_csv_to_df(path):
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(path, encoding='euc-kr')
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding='latin1')
        
        print(f"로드된 CSV 파일 경로: {path}")
        print(f"데이터 샘플:\n{df.head()}")
        print(f"총 행 수: {len(df)}")
    
        csv_str = df.to_csv(index=False)
        return csv_str


    # 각 에이전트별 CSVSearchTool 초기화
    travel_csv_tool = CSVSearchTool(csv=load_csv_to_df(csv_paths['travel']) if csv_paths['travel'] else None)
    food_csv_tool = CSVSearchTool(csv=load_csv_to_df(csv_paths['food']) if csv_paths['food'] else None)

    # 맞춤형 여행 조사 에이전트
    personal_researcher = Agent(
        role='맞춤형 여행 조사 에이전트',
        goal=f'{user_info["age"]} {user_info["gender"]}의 맞춤형 여행지 추천',
        backstory=f"""여행 전문가로서 {user_info['age']} {user_info['gender']}이(가) {user_info['companion']}와 
                   함께하는 {user_info['style']} 스타일의 여행을 위한 최적의 장소들을 추천합니다.""",
        tools=[search_tool],
        llm=llm,
        verbose=True
    )

    # 1. 관광지 분석 Agent
    tourist_spot_researcher = Agent(
        role='Tourist Spot Analyst',
        goal=f'{user_info["style"]} 스타일에 맞는 관광지 분석',
        backstory=f'{destination}관광지 데이터를 분석하여 {user_info["style"]} 스타일에 적합한 장소를 추천하는 전문가입니다.',
        tools=[CSVSearchTool(csv=load_csv_to_df(csv_paths['travel']))],

        verbose=True
    )


    # 2. 맛집 분석 Agent
    restaurant_researcher = Agent(
        role='Restaurant Analyst',
        goal='{destination}관광지 주변 맛집 분석',
        backstory='{destination}관광지 주변의 맛집을 분석하여 적합한 식당을 추천하는 전문가입니다.',
        tools=[CSVSearchTool(csv=load_csv_to_df(csv_paths['food']))],
        verbose=True
    )


    # 일정 계획 에이전트
    itinerary_planner = Agent(
        role='여행 일정 수립 에이전트',
        goal='효율적인 여행 동선 계획',
        backstory="""personal_task에서 추천된 {style} 장소들을 중심으로 {days}일간의 여행 일정을 계획해주세요.
                    1시간 이내 이동 가능한 효율적인 동선을 설계하는 전문가입니다.""",
        tools=[local_tool],
        llm=llm,
        verbose=True
    )

    return personal_researcher, tourist_spot_researcher, restaurant_researcher, itinerary_planner


def create_tasks(agents, user_info):
    personal_researcher, tourist_spot_researcher, restaurant_researcher, itinerary_planner = agents
    destination = user_info['destination']
    style = user_info['style'] 
    age = user_info['age'] 

    search_queries = {
        '국가유산': f"{destination} {age} 추천 유적지 문화재 박물관 명소",
        '휴양': f"{destination} {age} 추천 힐링스팟 카페 휴식 산 공원 명소",
        '액티비티': f"{destination} {age} 추천 액티비티 체험 관광 즐길거리",
        '식도락': f"{destination} {age} 맛집 추천 현지맛집 유명식당",
        'SNS감성': f"{destination} {age} 인스타 핫플레이스 감성카페 포토스팟"
    }
    
    travel_style_prompts = {
        '국가유산': f"""
            {destination}의 대표적인 국가유산와 역사 관광지를 찾아주세요.
            - 유명 국가유산와 유적지
            - 박물관과 전시관
            - {user_info['companion']}와 함께 둘러보기 좋은 곳
            - 관람 소요시간과 볼거리 포함
        """,
        
        '휴양': f"""
            {age}연령대가 {destination}의 힐링하기 좋은 장소들을 찾아주세요.
            - 힐링 명소와 조용한 장소
            - 경관이 좋은 카페와 휴식 공간
            - 자연 경관이 아름다운 곳
            - {user_info['companion']}와 편안한 시간을 보내기 좋은 곳
        """,
        
        '액티비티': f"""
            {age}연령대가 {destination}의 체험형 관광지와 액티비티를 찾아주세요.
            - {user_info['age']} {user_info['gender']}의 체력 수준에 적합한 활동
            - {user_info['companion']}와 함께 즐기기 좋은 체험
            - 안전하고 초보자도 할 수 있는 활동
            - 계절/날씨별 추천 활동
        """,
        
        '식도락': f"""
            {destination}의 맛집과 음식점을 찾아주세요.
            - 현지 맛집과 유명 식당
            - {user_info['companion']}와 식사하기 좋은 분위기의 장소
            - 특별한 지역 음식과 대표 메뉴
            - 가격대와 영업시간 정보
        """,
        
        'SNS감성': f"""
            {destination}의 인스타그램 핫플레이스를 찾아주세요.
            - 인기 있는 포토스팟
            - 뷰가 좋은 감성 카페
            - {user_info['age']} {user_info['gender']}이 좋아할만한 트렌디한 장소
            - 예쁜 사진을 찍을 수 있는 명소
        """
    }


    
    # search_query = f"{destination} {style}추천 {user_info['age']} {user_info['gender']} {user_info['companion']}"


    personal_task = Task(
        name="사용자 맞춤형 여행 조사",
        description=f"""
            Search Query: {search_queries[style]}

            다음 프롬프트를보고 연령대 여행스타일 맞춤 장소 15개 리스트를 작성하세요.
            {travel_style_prompts[style]}
            
            **주의사항:**
            - {user_info['age']} {user_info['gender']}이(가) {user_info['companion']}와 함께하는 
            {user_info['style']} {destination} 여행을 위한 장소 15개 리스트를 작성하세요.
            - 계절과 날씨를 고려한 추천
            - {user_info['age']} {user_info['gender']}의 선호도 고려

            반드시 문자열로 작성

            """,
        expected_output='사용자 특성에 맞는 맞춤형 여행 추천 보고서',
        agent=personal_researcher
    )


    # Task 1: 관광지 분석
    tourist_spot_task = Task(
        name="관광지 데이터 분석",
        description=f"""
            {destination}
            반드시 CSV 파일의 '분류' 컬럼에서 '{style}' 스타일에 맞는 장소만 찾아주세요.
            - 여행 스타일별 키워드:
            * 문화재: '역사유적지', '박물관', '전시시설'
            * 휴양: '자연경관', '도시공원', '테마공원', '레저스포츠시설'
            * 액티비티: '레저스포츠시설', '체험시설'
            * SNS감성: '랜드마크관광', '테마공원'
            * 식도락: '시장', '쇼핑몰'
        
            다음 형식으로 출력해주세요:
            장소: [관광지명]
            주소: [주소]
    
        """,
        expected_output="관광지 추천 목록",
        agent=tourist_spot_researcher
    )


        # Task 2: 맛집 분석
    restaurant_task = Task(
        name="{destination}주변 맛집 분석",
        description=f"""
                {destination}
                tourist_spot_task에서 조회된 관광지 주소를 기반으로  파일에서 주변 맛집을 검색하세요.
                반드시 CSV 파일의 '주소' 컬럼에서 행정구역이 일치하는 장소만 찾아주세요.
                
                각 구별로 다음과 같이 한 번씩 검색하세요:
                {{
                "search_query": "행정구", "행정시"
                }}

                이런 식으로 각 시,구별로 개별 검색을 수행하세요.

                  
                각 관광지 주소의 구(區)를 기준으로 2-3곳의 맛집을 추천하세요.

                다음 형식으로 출력해주세요.
                1. 식당: [맛집명]
                주소: [도로명주소]

            """,
        expected_output="행정구역별 맛집 추천 목록",
        agent=restaurant_researcher
)


    nights, days = calculate_trip_days(user_info['start_date'], user_info['end_date'])
    
    

    planning_task = Task(
        name="여행 일정 계획 수립",
        description=f"""
                도로명주소를 문자열로 전달하고 반환하세요.
                세 가지 task의 결과를 균형있게 활용하여 {days}일간의 {age}대 {style} {destination}여행 일정을 계획하세요.

                 장소 주소 검색 시:
                - 네이버 검색은 **다음과 같은 형식**으로 장소명을 전달해야 합니다.
                **네이버 검색 사용 시 주의사항:**
                - Action Input은 반드시 딕셔너리 형식으로 입력하세요.
                - 올바른 형식: **Action Input: {{"query": "장소명"}}**
                - 잘못된 형식:
                - Action Input: 가로수길  # 문자열만 입력하면 안 됩니다.
                - Action Input: "가로수길"  # 따옴표로 감싼 문자열만 입력하면 안 됩니다.
                - Action Input: {{"name": "가로수길"}}  # 키 이름이 잘못되었습니다.

                반영 비율:
                1. personal_task (웹 검색 결과) - 60% 반영
                    - 반드시 하루에 2-3곳은 포함할 것

                2. tourist_spot_task (관광지 CSV) - 20% 반영
                    - 유명 관광지나 랜드마크는 하루 1곳 정도만 포함
                    - 이동 동선 상 필요한 경우에만 추가
           
                3. restaurant_task (맛집 CSV) - 20% 반영
                    - 점심, 저녁 식사 시간에 맞춰 배치
                    - 주요 장소 근처의 맛집 위주로 선정
                

                **일정 작성 가이드:**
                이동 시간 규칙:
                1. 1시간 이내로 이동 장소
                2. 연속된 장소들은 반드시 같은 시/군/구 내에서 선택
                3. 다른 시/군으로 이동할 경우 다음 날 일정으로 계획
                4. 하루에 한 개의 시/군만 방문
                5. 각 장소의 도로명주소 필수 (네이버 검색으로 확인)
                6. 반드시 식사, 간식, 휴식 등을 고려해 현실적인 여행계획을 고려하세요.
                 - 오전(9-12시): personal_task + tourist_spot_task
                - 점심(12-2시): restaurant_task의 맛집
                - 오후(2-6시): tourist_spot_task의 관광지 + personal_task의 장소
                - 저녁(6시 이후): restaurant_task의 맛집 + personal_task의 저녁 장소

                1일차: [도시/군/구] 내 일정만 구성
                2일차: [도시/군/구] 내 일정만 구성
                3일차: [도시/군/구] 내 일정만 구성

                **반드시 아래의 JSON 형식으로 작성하고, {days}일 모두 포함해야 합니다**

        {{
            "result": {{
                "Day 1": [
                    {{
                        "time": "시간",
                        "place": {{
                            "장소": "장소명",
                            "address": "주소"
                        }}
                    }},
                    {{
                        "time": "시간",
                        "place": {{
                            "장소": "장소명",
                            "address": "주소"
                        }}
                    }},
                    {{
                        "time": "시간",
                        "place": {{
                            "장소": "장소명",
                            "address": "주소"
                        }}
                    }}
                ],
                "Day 2": [
                    ... (다음날 일정도 동일한 형식으로 반복)
                ]
            }}
        }}

        **중요:**
        - 오직 JSON 데이터만 출력하세요.
        - 불필요한 설명이나 추가 텍스트를 포함하지 마세요.
        - JSON 형식을 엄격하게 지켜주세요.
        

        [다음날 일정도 동일한 형식으로 반복]
    
           
        """,
        expected_output="정확한 형식의 {days}일간 여행 일정표",
        agent=itinerary_planner,
        
    )

    return [personal_task, tourist_spot_task, restaurant_task,  planning_task]



def plan_travel(user_info: dict):
    from langchain_openai import ChatOpenAI

   # LLM 설정
    llm = ChatOpenAI(
       api_key=os.getenv("OPENAI_API_KEY"),
       model_name="gpt-4o-mini",
       temperature=0.7,
       max_tokens=2000
    )


    # 날짜를 문자열로 변환하여 user_info에 저장
    user_info['start_date'] = user_info['start_date'].strftime('%Y-%m-%d')
    user_info['end_date'] = user_info['end_date'].strftime('%Y-%m-%d')


   # 에이전트 생성
    personal_researcher, tourist_spot_researcher, restaurant_researcher, itinerary_planner = create_travel_agents(llm, user_info)
   
   # 작업 생성 
    tasks = create_tasks([personal_researcher, tourist_spot_researcher, restaurant_researcher, itinerary_planner], user_info)
    personal_task = tasks[0]
    tourist_spot_task = tasks[1]
    restaurant_task = tasks[2]
    planning_task = tasks[3]

    crew = Crew(
        agents=[personal_researcher, tourist_spot_researcher, restaurant_researcher, itinerary_planner],
        tasks=[tourist_spot_task, restaurant_task, personal_task, planning_task],
        verbose=True,
        task_dependencies={
              # 맛집은 관광지 기반으로 검색
            planning_task: [personal_task]   # planning은 모든 결과 활용
        }
    )
    # 시작 전 Crew의 설정 상태를 출력
    print("Crew 설정 상태:", crew)

    crew_output = crew.kickoff()

    # planning_task의 인덱스 찾기
    try:
        task_index = crew.tasks.index(planning_task)
    except ValueError:
        print("Error: planning_task가 crew.tasks에 없습니다.")
        return None

    # planning_task의 출력 가져오기
    planning_task_output = crew_output.tasks_output[task_index]

    # 결과 추출
    result = None
    if hasattr(planning_task_output, 'raw'):
        result = planning_task_output.raw
    elif hasattr(planning_task_output, 'summary'):
        result = planning_task_output.summary
    elif hasattr(planning_task_output, 'dict'):
        result_dict = planning_task_output.dict()
        if 'raw' in result_dict:
            result = result_dict['raw']
        elif 'summary' in result_dict:
            result = result_dict['summary']
    else:
        print("Error: planning_task_output에서 결과를 추출할 수 없습니다.")
        print("TaskOutput 객체의 속성:", dir(planning_task_output))
        return None

    # # 결과 반환
    # return result

    # 결과 반환
    if isinstance(result, str):
        # 이미 JSON 문자열인 경우
        return json.loads(result)
    elif isinstance(result, dict):
        # 딕셔너리인 경우
        return result
    else:
        print("Error: 예상치 못한 result 형식:", type(result))
        return None




if __name__ == "__main__":
    user_info = {
       "gender": "남성",
       "age": "50대",
       "companion": "친구",
       "destination": "제주",
       "style": "휴양",
       "start_date": "2024-10-30",
       "end_date": "2024-11-1"
    }
   
    # base_path = '../data'
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'travel', 'data')

    if os.path.exists(base_path):
        print("경로가 존재합니다.")
    else:
        print("경로를 찾을 수 없습니다.")

    result = plan_travel(user_info)

    if result is not None:
        print("\n=== 최종 여행 계획 ===")
        import json

        try:
            formatted_result = json.loads(result)
            # formatted_result를 사용하여 원하는 데이터 처리
            print(json.dumps(formatted_result, ensure_ascii=False, indent=2))
        except json.JSONDecodeError as e:
            print("JSON 파싱 오류:", e)
            print("에이전트의 출력 결과를 확인하세요.")
            print(result)
    else:
        print("여행 일정 생성 중 오류가 발생했습니다.")