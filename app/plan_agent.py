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
from functools import lru_cache  # lru_cache만 직접 import하는 것은 잘못된 방식입니다

import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        # 'travel': os.path.join(TRAVEL_DATA_DIR, 'travel'),
        # 'food': os.path.join(TRAVEL_DATA_DIR, 'food'),
        'history': os.path.join(TRAVEL_DATA_DIR, 'history'),
        
    }
    # result = {'travel': None, 'food': None, 'history': None}

    result = {'history': None}
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


def load_csv_to_df(file_path: str) -> pd.DataFrame:
    """CSV 파일을 DataFrame으로 안전하게 로드"""
    try:
        if isinstance(file_path, pd.DataFrame):  # 이미 DataFrame인 경우
            return file_path
        
        if not file_path or not os.path.exists(file_path):
            logger.warning(f"CSV 파일이 존재하지 않습니다: {file_path}")
            return pd.DataFrame()

        # 파일 읽기 시도
        df = None
        encodings = ['utf-8', 'euc-kr', 'cp949']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"CSV 파일을 {encoding} 인코딩으로 성공적으로 로드했습니다")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"{encoding} 인코딩으로 로드 실패: {str(e)}")
                continue
        
        if df is None:
            logger.error("모든 인코딩으로 CSV 파일 로드 실패")
            return pd.DataFrame()
            
        # 데이터 확인 로깅
        logger.info(f"로드된 CSV 파일 경로: {file_path}")
        logger.info(f"데이터 샘플:\n{df.head()}")
        logger.info(f"총 행 수: {len(df)}")
        
        return df

    except Exception as e:
        logger.error(f"CSV 파일 로드 중 예외 발생: {str(e)}")
        return pd.DataFrame()


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
    

class NaverBlogSearchTool(BaseTool):
    """네이버 블로그 검색 도구"""
    name: str = "Naver Blog Search"
    description: str = "네이버 블로그에서 여행 정보와 실제 방문자 후기를 검색합니다."
    client_id: str = ""
    client_secret: str = ""
    headers: dict = {}

    def __init__(self):
        super().__init__()
        self.client_id = os.getenv("NAVER_CLIENT_ID")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET")
        self.headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        # 캐시된 검색 메서드 초기화
        self._cached_search = lru_cache(maxsize=100)(self._search)

    def _search(self, query: str) -> str:
        """실제 검색을 수행하는 내부 메서드"""
        url = "https://openapi.naver.com/v1/search/blog"
        full_query = f"{query} 여행 후기"
        params = {
            "query": full_query,
            "display": 10,
            "sort": "sim"
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            items = response.json().get('items', [])
            results = "📝 블로그 검색 결과:\n"
            for item in items:
                results += f"""
제목: {item['title'].replace('<b>', '').replace('</b>', '')}
내용: {item['description'].replace('<b>', '').replace('</b>', '')}
링크: {item['link']}
작성일: {item.get('postdate', '정보 없음')}
-------------------"""
            return results
        return "검색 결과를 찾을 수 없습니다."

    def _run(self, query: str) -> str:
        """BaseTool 인터페이스를 구현하는 실행 메서드"""
        try:
            if isinstance(query, dict) and 'query' in query:
                query = query['query']
            return self._cached_search(query)
        except Exception as e:
            return f"검색 중 오류가 발생했습니다: {str(e)}"


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
    # 도구 초기화
    search_tool = SerperDevTool()
    local_tool = NaverLocalSearchTool()
    blog_tool = NaverBlogSearchTool()

    
    destination = user_info["destination"]
    detail_destination = user_info.get("detail_destination", "")
    style = user_info["style"]
    age = user_info["age"]
    
    csv_paths = get_csv_file_paths(destination)

    
    # CSV 도구 초기화
    history_csv_tool = None
    if csv_paths.get('history'):
        try:
            df = load_csv_to_df(csv_paths['history'])
            if not df.empty:
                # DataFrame을 문자열로 변환
                csv_str = df.to_csv(index=False)
                history_csv_tool = CSVSearchTool(csv=csv_str)
        except Exception as e:
            logger.error(f"History CSV 로드 실패: {str(e)}")



    # 1. 스타일별 관광지 분석 에이전트
    style_configs = {
        '국가유산': {
            # 'tools': [CSVSearchTool(csv=load_csv_to_df(csv_paths['history'])) if csv_paths else None],
            'tools': [history_csv_tool] if history_csv_tool else [search_tool],
            'backstory': f'{destination}{detail_destination}의 역사문화 데이터를 분석하여 문화재, 박물관, 전시시설을 추천하는 전문가입니다.'
        },
        '휴양': {
            # 'tools': [CSVSearchTool(csv=load_csv_to_df(csv_paths['travel'])) if csv_paths else None],
            'tools': [history_csv_tool] if history_csv_tool else [search_tool],
            'backstory': f'{destination}{detail_destination}의 관광 데이터를 분석하여 자연경관, 공원, 휴식 공간을 추천하는 전문가입니다.'
        },
        '액티비티': {
            'tools': [search_tool],
            'query': f"'{user_info['destination']} {user_info['detail_destination']} 체험 레저스포츠'",
            'backstory': f'{destination}{detail_destination}의 체험형 관광지와 레저스포츠 시설을 추천하는 전문가입니다.'
        },
        'SNS감성': {
            'tools': [blog_tool],
            'query': f"'{user_info['destination']} {user_info['detail_destination']} 인스타 핫플 포토스팟'",
            'backstory': f'{destination}{detail_destination}의 인스타그램 핫플레이스와 포토스팟을 추천하는 트렌드 전문가입니다.'
        },
        '식도락': {
            'tools': [search_tool],
            'query': f"'{user_info['destination']} {user_info['detail_destination']} 현지맛집, 식도락'",
            'backstory': f'{destination}{detail_destination}의 맛집과 식도락 여행지를 추천하는 음식 전문가입니다.'
        }
    }

    
    
    tourist_spot_researcher = Agent(
        role=f'{style} 관광지 전문가',
        goal=f'{destination} {detail_destination}의 {style} 특화 관광지 분석',
        backstory=style_configs[style]['backstory'],
        tools=style_configs[style]['tools'],
        llm=llm,
        verbose=True
    )

    # 2. 맛집 분석 에이전트 (구글 검색 기반)
    restaurant_researcher = Agent(
        role='맛집 분석 전문가',
        goal=f'{destination} {detail_destination}의 tourist_spot_researcher 여행장소 근처 맛집 분석',
        backstory=f'tourist_spot_researcher 여행장소 근처 맛집을 검색하고 분석하여 최적의 식당을 추천하는 전문가입니다.',
        tools=[search_tool],
        llm=llm,
        verbose=True
    )

    # 3. 일정 계획 에이전트
    itinerary_planner = Agent(
        role='여행 일정 계획가',
        goal='효율적인 여행 동선 설계',
        backstory=f"""
            {style} 스타일의 관광지와 맛집을 연계하여 {user_info['age']} {user_info['gender']}에게 
            최적화된 일정을 계획하는 전문가입니다. 대중교통과 도보 이동을 고려하여 
            30분 이내 이동 가능한 효율적인 동선을 설계합니다.
        """,
        tools=[local_tool],
        llm=llm,
        verbose=True
    )

    return tourist_spot_researcher, restaurant_researcher, itinerary_planner

def create_tasks(agents, user_info):
    tourist_spot_researcher, restaurant_researcher, itinerary_planner = agents

    style = user_info['style'] 
    age = user_info['age'] 
    destination = user_info['destination']
    detail_destination = user_info['detail_destination']
    
    style_task_configs = {
        '국가유산': {
            'query': f"'{user_info['destination']} {user_info['detail_destination']} 해당하는 문화재 박물관 역사유적지 전시관'",
            'focus': """
                - 역사적 가치가 있는 문화재와 유적지
            """
        },
        '휴양': {
            'query': f"'{user_info['destination']} {user_info['detail_destination']} 자연 공원 힐링 명소'",
            'focus': """
                - 자연경관이 뛰어난 장소
                - 도시공원과 휴식공간
                - 산책로와 전망대
                - 힐링 카페와 휴식 공간
            """
        },
        '액티비티': {
            'query': f"'{user_info['destination']} {user_info['detail_destination']} 체험 액티비티 레저'",
            'focus': """
                - 체험형 관광지와 액티비티
                - 레저스포츠 시설
            
            """
        },
        'SNS감성': {
            'query': f"'{user_info['destination']} {user_info['detail_destination']} 인스타 핫플 포토스팟'",
            'focus': """
                - 인스타그램 인기 장소
                - 뷰가 좋은 카페와 레스토랑
                - 포토스팟과 촬영 포인트
            
            """
        },
        '식도락': {
            'query': f"'{user_info['destination']} {user_info['detail_destination']} 맛집 현지맛집 먹거리'",
            'focus': """
                - 현지 맛집과 대표 음식점
                - 특색있는 카페와 디저트
                - 예약 필요 여부와 웨이팅
                - 인기 메뉴와 가격대
            """
        }
    }

    config = style_task_configs[user_info['style']]

    # 1. 관광지 분석 태스크
    spot_analysis_task = Task(
        name=f"{user_info['style']} 관광지 분석",
        description=f"""
            검색어: {config['query']}
            
            {user_info['style']} 스타일에 맞는 장소를 찾아 분석해주세요:
            {config['focus']}
            
            - {user_info['age']} {user_info['gender']}의 선호도 고려
            - {user_info['companion']}와 함께하기 좋은 장소 위주
            - 상세 주소와 영업시간 포함
        """,
        expected_output=f"{user_info['destination']} {user_info['style']} 한국어 관광지 목록",  # expected_output 추가

        agent=tourist_spot_researcher
    )

    # 2. 맛집 분석 태스크
    restaurant_task = Task(
        name="주변 맛집 분석",
        description=f"""
            spot_analysis_task 에서 찾은 관광지마다 주변 맛집과 카페를 검색하세요.
            관광지 주변의 맛집을 검색하고 분석해주세요:
            - 관광지에서 도보 20분 이내 거리
            - {user_info['age']} {user_info['gender']}의 선호도 고려
            - {user_info['companion']}와 식사하기 좋은 분위기
            
        """,
        expected_output="한국어로 된 관광지 주변 맛집 목록",  # expected_output 추가

        agent=restaurant_researcher
    )


    # 맛집 검색 쿼리 설정
    search_query = f"{user_info['destination']} {user_info['detail_destination']} 맛집"

    tool = SerperDevTool()

    try:
        # 맛집 검색 수행
        restaurant_results = tool._run({"tool_input": search_query})
    
        # 결과를 restaurant_task에 전달
        restaurant_task.description += f"\n\n검색된 맛집 정보:\n{restaurant_results}"
    except Exception as e:
        print(f"맛집 검색 중 오류 발생: {e}")

    nights, days = calculate_trip_days(user_info['start_date'], user_info['end_date'])



    # 3. 일정 계획 태스크
    planning_task = Task(
        name="여행 일정 계획 수립",
        description=f"""
                도로명주소를 문자열로 전달하고 반환하세요.
                tourist_spot_researcher결과값과 restaurant_researcher 조사 결과를 바탕으로 이동 동선을 짜주세요
                관광지 근처 맛집들을 일정에 반영하여 하루 동안 관광지와 맛집을 함께 고려해 주세요.
                {days}일간의 {age}대 {style} {destination}{detail_destination}여행 일정을 계획하세요.

                 장소 주소 검색 시:
                - 네이버 검색은 **다음과 같은 형식**으로 장소명을 전달해야 합니다.
                **네이버 검색 사용 시 주의사항:**
                - Action Input은 반드시 딕셔너리 형식으로 입력하세요.
                - 올바른 형식: **Action Input: {{"query": "장소명"}}**
                - 잘못된 형식:
                - Action Input: 가로수길  # 문자열만 입력하면 안 됩니다.
                - Action Input: "가로수길"  # 따옴표로 감싼 문자열만 입력하면 안 됩니다.
                - Action Input: {{"name": "가로수길"}}  # 키 이름이 잘못되었습니다.



                **일정 작성 가이드: 여행장소와 맛집은 반드시 동일한 행정구역 안에서 추천하세요, 한번 일정에 쓰인곳은 다시 사용하지마세요**
                 - **이동 시간 규칙:**
                    하루의 일정
                    1. 한번 일정에 사용한 장소로 다시 일정을 짜지 마세요.
                    2. 여행장소와 여행장소 근처의 맛집을 붙여서 경로를 짜주세요.
                    3. 여행장소끼리는 30분을 넘지 않도록 해주세요.
                    4. 각 장소의 도로명주소를 반드시 확인하고, 네이버 검색을 통해 정확한 주소를 입력해 주세요.
                    5. 식사, 간식, 휴식 등을 고려하여 현실적인 여행 계획을 세워주세요.

                **일정 시간대 규칙:**
                - 오전 (9-12시): 관광지 방문 (tourist_spot_task) 
                - 점심 (12-2시): 맛집 방문 (오전의 도로명과동일한 "동", "읍","로", "길" 의 맛집) 
                - 오후 (2-4시): 관광지  (tourist_spot_task )
                - 오후 (3-5시): 카페 방문( 오후의 도로명과동일한 "동", "읍","로", "길" 의 맛집)
                - 저녁 (6시 이후): 맛집 방문 (restaurant_task) 또는 관광지 (tourist_task)

            

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
        expected_output="정확한 형식의 {days}일간 한국어 여행 일정표",
        agent=itinerary_planner,
        
    )

    return [spot_analysis_task, restaurant_task, planning_task]







def plan_travel(user_info: dict):
    from langchain_openai import ChatOpenAI

   # LLM 설정
    llm = ChatOpenAI(
       api_key=os.getenv("OPENAI_API_KEY"),
       model_name="gpt-4o-mini",
       temperature=0.7,
       max_tokens=2000,
       model_kwargs={
            "messages": [
                {"role": "system", "content": "당신은 한국어로 응답하는 여행 계획 전문가입니다. 모든 응답은 한국어로 해주세요."},
            ]}
    )

   # 에이전트 생성
    tourist_spot_researcher, restaurant_researcher, itinerary_planner = create_travel_agents(llm, user_info)
   
   # 작업 생성 
    tasks = create_tasks([tourist_spot_researcher, restaurant_researcher, itinerary_planner], user_info)
    spot_analysis_task = tasks[0]
    restaurant_task = tasks[1]
    planning_task = tasks[2]  # 3개의 task만 있으므로 인덱스는 0, 1, 2

    crew = Crew(
        agents=[tourist_spot_researcher, restaurant_researcher, itinerary_planner],
        tasks=[spot_analysis_task, restaurant_task, planning_task],
        verbose=True,
        task_dependencies={
            
            restaurant_task: [spot_analysis_task],  # 맛집 분석은 관광지 분석 결과에 의존
            planning_task: [spot_analysis_task, restaurant_task] 
        }
    )
    # crew_output = crew.kickoff()

    # 첫 번째 crew 실행
    try:
        # crew 한 번만 실행
        crew_output = crew.kickoff()
        
        # 관광지 분석 결과 처리
        tourist_spot_analysis_result = None
        if crew_output and crew_output.tasks_output:
            raw_output = crew_output.tasks_output[0].raw
            if raw_output:
                if isinstance(raw_output, (list, dict)):
                    tourist_spot_analysis_result = json.dumps(raw_output, ensure_ascii=False)
                else:
                    tourist_spot_analysis_result = str(raw_output)
            else:
                tourist_spot_analysis_result = "분석 결과가 없습니다"
        
        if not tourist_spot_analysis_result:
            tourist_spot_analysis_result = f"{user_info['destination']} {user_info['detail_destination']}"
        
        logger.info(f"관광지 분석 결과: {tourist_spot_analysis_result}")

        # restaurant_task 설명 업데이트
        search_query = f"{tourist_spot_analysis_result} 주변 맛집"
        restaurant_task.description = f"""
            관광지 정보: {tourist_spot_analysis_result}
            해당 정보를 바탕으로, 주변의 맛집을 검색하고 추천해주세요. 
            검색 쿼리: {search_query}
        """


    # planning_task의 인덱스 찾기
        try:
            task_index = crew.tasks.index(planning_task)
        except ValueError:
            print("Error: planning_task가 crew.tasks에 없습니다.")
            return None

        # planning_task의 출력 가져오기
        planning_task_output = crew_output.tasks_output[task_index]

        # TaskOutput에서 raw 데이터 추출
        raw_output = planning_task_output.raw

        # data 디렉토리 생성 및 파일 저장
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'travel','data')
        os.makedirs(data_dir, exist_ok=True)
        output_file = os.path.join(data_dir, 'test.tmp')
    
        # planning_task_output을 직접 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            if isinstance(raw_output, str):
                f.write(raw_output)
            else:
            # raw_output이 dict나 list 같은 객체인 경우 JSON으로 변환
                json.dump(raw_output, f, ensure_ascii=False, indent=2)
    
        print(f"결과가 {output_file}에 저장되었습니다.")


        # 결과 추출
        # result = None
        # if hasattr(planning_task_output, 'raw'):
        #     result = planning_task_output.raw
        # elif hasattr(planning_task_output, 'summary'):
        #     result = planning_task_output.summary
        # elif hasattr(planning_task_output, 'dict'):
        #     result_dict = planning_task_output.dict()
        #     if 'raw' in result_dict:
        #         result = result_dict['raw']
        #     elif 'summary' in result_dict:
        #         result = result_dict['summary']
        # else:
        #     print("Error: planning_task_output에서 결과를 추출할 수 없습니다.")
        #     print("TaskOutput 객체의 속성:", dir(planning_task_output))
        #     return None
        # test.tmp 파일 읽기
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
                # JSON으로 변환
                result = json.loads(file_content)
        except Exception as e:
            logger.error(f"test.tmp 파일 읽기 중 오류 발생: {str(e)}")
        # 결과 반환
        return result
    
    

    except Exception as e:
        logger.error(f"Crew 실행 중 오류 발생: {str(e)}")
        return None


if __name__ == "__main__":
    user_info = {
       "gender": "남성",
       "age": "50대",
       "companion": "친구",
       "destination": "제주",
       "detail_destination": "제주",
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

    