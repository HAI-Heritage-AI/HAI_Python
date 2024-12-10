from langchain_openai import ChatOpenAI  # OpenAI의 GPT 모델을 사용하기 위한 클래스
from typing import Optional  # 타입 힌팅을 위한 Optional 타입
import os  # 환경 변수 및 파일 경로 관련 작업
import requests  # HTTP 요청을 위한 라이브러리
from dotenv import load_dotenv  # .env 파일에서 환경 변수를 로드
from app.plan_agent import plan_travel, calculate_trip_days  # 여행 계획 관련 함수들
import json  # JSON 데이터 처리
import pandas as pd  # 데이터 프레임 처리
from pathlib import Path  # 파일 경로 처리
from geopy.geocoders import Nominatim  # 주소를 좌표로 변환
from geopy.distance import geodesic    # 두 좌표 간 거리 계산

# .env 파일에서 환경 변수 로드
load_dotenv()

class TravelChatAgent:
    def __init__(self):
        # 기본 속성들 초기화
        self.current_travel_plan = None  # 현재 여행 계획 저장
        self.destination = None  # 주요 여행 목적지
        self.detail_destination = None  # 세부 여행 목적지
        self.travel_style = None  # 여행 ��타일
        self.user_info = None  # 사용자 정보 저장
        
        # test.tmp 파일에서 여행 플랜 로드
        plan_data = self.load_travel_plan_from_file()
        if plan_data:
            self.current_travel_plan = plan_data
            # 여행 플랜에서 목적지 정보들을 추출하여 설정
            self.destination = plan_data.get('destination')  # 주요 목적지 (예: 도/시 단위)
            self.detail_destination = plan_data.get('detail_destination')  # 세부 목적지 (예: 구/군/시 단위)
            print(f"목적지가 {self.destination} {self.detail_destination}로 설정되었습니다.")
        
        # GPT 모델 초기화
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),  # OpenAI API 키
            model_name="gpt-3.5-turbo",  # 사용할 모델
            temperature=0.7  # 응답의 창의성 조절 (0: 보수적, 1: 창의적)
        )
        self.chat_history = []  # 대화 기록 저장 리스트
        self.max_turns = 6  # 최대 대화 턴 수 (6턴 = 질문6개 + 답변6개)
        
        # 네이버 API 헤더 설정
        self.naver_headers = {
            "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
            "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")
        }

        # CSV 데이터 파일들 로드
        self.data_path = Path("app/travel/data")  # 데이터 파일 기본 경로
        self.travel_data = self._load_csv_data("travel")  # 관광지 데이터
        self.food_data = self._load_csv_data("food")  # 음식점 데이터
        self.festival_data = self._load_csv_data("festival")  # 축제 데이터

    def set_user_info(self, user_info: dict):
        """사용자 정보를 설정하는 메서드"""
        try:
            self.user_info = user_info  # 전체 사용자 정보 저장
            self.destination = user_info.get('destination')  # 목적지 설정
            self.travel_style = user_info.get('style')  # 여행 스타일 설정
            self.current_travel_plan = user_info.get('travel_plan', {})  # 여행 계획 설정
        except Exception as e:
            print(f"여행자 정보 설정 중 오류 발생: {e}")
            self.current_travel_plan = {}

    def _parse_travel_plan(self, travel_plan: dict) -> dict:
        """여행 플랜에서 주요 정보를 추출하는 메서드"""
        plan_info = {
            'places': [],  # 계획된 장소들의 리스트
            'schedule': {}  # 일정별 정보를 담는 딕셔너리
        }
        
        try:
            # 여행 플랜이 없거나 travel_plan 키가 없는 경��� 기본값 반환
            if not travel_plan or 'travel_plan' not in travel_plan:
                return plan_info
            
            # 일자별 일정 정보 추출
            days_schedule = travel_plan['travel_plan']
            
            # 각 일자별로 활동 정보 처리
            for day, activities in days_schedule.items():
                day_info = []  # 해당 일자의 장소 정보를 담을 리스트
                for activity in activities:
                    # 장소 정보가 있는 경우만 처리
                    if 'place' in activity and '장소' in activity['place']:
                        place = {
                            'name': activity['place']['장소'],  # 장소명
                            'address': activity['place']['address'],  # 주소
                            'time': activity.get('time', '')  # 방문 시간 (없으면 빈 문자열)
                        }
                        day_info.append(place)  # 일자별 정보에 추가
                        plan_info['places'].append(place)  # 전체 장소 목록에 추가
                
                # 해당 일자에 장소가 있는 경우만 저장
                if day_info:
                    plan_info['schedule'][day] = day_info
                
        except Exception as e:
            print(f"여행 플랜 파싱 중 오류 발생: {e}")
        
        return plan_info


    def search_naver_blog(self, query: str) -> str:
        """네이버 블로그 검색 - 지역 필터링 추가"""
        if not self.destination:
            print("Warning: destination이 설정되지 않았습니다!")
            return "여행 목적지 정보가 없습니다."

        # 네이버 블로그 검색 API URL
        url = "https://openapi.naver.com/v1/search/blog"
        
        print(f"현재 설정된 destination: {self.destination}")
        
        # 검색어에 목적지를 앞에 추가하여 정확도 향상
        search_query = f"{self.destination} {query}"
        params = {
            "query": search_query,  # 검색할 키워드
            "display": 10,  # 검색 결과 개수
            "sort": "sim"   # 정렬 기준 (sim: 정확도순)
        }
        
        print(f"\n=== 네이버 블로그 검색 요청 ===")
        print(f"검색어: {search_query}")
        
        # API 요청 보내기
        response = requests.get(url, headers=self.naver_headers, params=params)
        print(f"응답 상태 코드: {response.status_code}")
        
        if response.status_code == 200:  # API 요청 성공
            items = response.json().get('items', [])
            filtered_items = []
            
            print(f"\n검색된 블로그 글 목록:")
            for item in items:
                print(f"\n제목: {item['title'].replace('<b>', '').replace('</b>', '')}")
                print(f"링크: {item['link']}")
                
                # 제목이나 내용에 목적지가 포함된 결과만 필터링하여 정확도 향상
                if self.destination in item['title'] or self.destination in item['description']:
                    filtered_items.append(item)
            
            if not filtered_items:
                return f"{self.destination}의 관련 정보를 찾을 수 없습니다."
            
            # 가장 관련성 높은 첫 번째 결과 선택하여 반환
            best_result = filtered_items[0]
            
            results = f"""
                가장 관련성 높은 블로그 글:
                제목: {best_result['title'].replace('<b>', '').replace('</b>', '')}
                내용: {best_result['description'].replace('<b>', '').replace('</b>', '')}
                링크: {best_result['link']}
                """
            return results
        return "검색 결과를 찾을 수 없습니다."

    def search_naver_local(self, query: str) -> str:
        """네이버 지역 검색 - ���역 필터링 추가"""
        # 네이버 지역 검색 API URL
        url = "https://openapi.naver.com/v1/search/local"
        
        # 검색 파라미터 설정
        params = {
            "query": f"{self.destination} {query}",  # 목적지를 포함한 검색어
            "display": 5,  # 검색 결과 개수
            "sort": "random"  # 정렬 방식 (random: 무작위)
        }
        
        # API 요청 보내기
        response = requests.get(url, headers=self.naver_headers, params=params)
        
        if response.status_code == 200:  # API 요청 성공
            items = response.json().get('items', [])
            filtered_items = []
            
            # 검색 결과 필터링
            for item in items:
                # 주소에 destination이 포함된 결과만 필터링하여 정확도 향상
                if self.destination in item['address']:
                    filtered_items.append(item)
            
            if not filtered_items:
                return f"{self.destination}의 관련 장소를 찾을 수 없습니다."
            
            # 검색 결과 포맷팅
            results = f"🏢 {self.destination} 관련 장소:\n"
            for item in filtered_items:
                results += f"""
                장소명: {item['title'].replace('<b>', '').replace('</b>', '')}
                주소: {item['address']}
                도로명: {item.get('roadAddress', '정보 없음')}
                카테고리: {item.get('category', '정보 없음')}
                전화: {item.get('telephone', '정보 없음')}
                -------------------"""
            return results
        return "검색 결과를 찾을 수 없습니다."

    async def get_answer(self, question: str, context: Optional[str] = None) -> str:
        """챗봇 답변 생성 - CSV 데이터 활용"""
        if not self.destination:
            return "죄송합니다. 여행 목적지 정보가 없습니다."
        
        # 요청한 개수 추출 (기본값 2)
        import re
        num_request = 2  # 기본값
        numbers = re.findall(r'(\d+)(?:곳|개)', question)  # "3곳", "5개" 등의 패턴 찾기
        if numbers:
            num_request = int(numbers[0])
        
        # 관광지 데이터 준비
        travel_data = None
        if self.destination in self.travel_data:
            # 요청한 개수만큼 랜덤 샘플링
            travel_data = self.travel_data[self.destination].sample(n=min(num_request, len(self.travel_data[self.destination])))
        
        # 음식점 데이터 준비
        food_key = f"{self.destination}맛집"
        food_data = None
        if food_key in self.food_data:
            # 사용자 질문에서 식당 이름 추출
            restaurant_name = None
            # "XX 갔다가" 패턴에서 식당 이름 추출
            if "갔다가" in question:
                restaurant_part = question.split("갔다가")[0]
                restaurant_name = restaurant_part.strip()

            # 기준 장소 찾기
            base_place = None
            if food_key in self.food_data and restaurant_name:
                # 식당 이름으로 데이터 검색
                base_place = self.food_data[food_key][
                    self.food_data[food_key]['관광지명'].str.contains(restaurant_name, case=False)
                ].iloc[0] if not self.food_data[food_key].empty else None

            # 근처 카페 찾기
            if base_place is not None:
                # 기준 장소 주변의 카페 검색
                nearby_cafes = self.find_nearby_places(
                    base_place['주소'], 
                    self.food_data[food_key][self.food_data[food_key]['분류'].str.contains('카페|찻집', case=False, na=False)]
                )
                food_data = nearby_cafes.head(num_request)
            else:
                # 기준 장소가 없으면 랜덤 추천
                food_data = self.food_data[food_key].sample(n=min(num_request, len(self.food_data[food_key])))

        # GPT 프롬프트 구성
        system_content = f"""당신은 {self.destination} 지역 전문 여행 챗봇입니다.

        현재 여행 계획:
        {json.dumps(self.current_travel_plan, ensure_ascii=False, indent=2)}

        사용자의 질문을 분석하여 관광지나 맛집 중 적절한 정보를 추천해주세요.

        사용 가능한 데이터:
        1. 관광지 데이터: {self._format_travel_results(travel_data) if travel_data is not None else '데이터 없음'}
        2. 맛집 데이터: {self._format_food_results(food_data) if food_data is not None else '데이터 없음'}

        이전 대화 기록:
        {self._format_chat_history()}

        규칙:
        1. 사용자의 질문 의도를 파악하여 관광지 또는 맛집을 추천해주세요.
        2. 제공된 데이터에서만 추천해주세요.
        3. 사용자가 요청한 개수({num_request}개)만큼 추천해주세요.
        4. 각 장소에 대해 주소와 특징을 함께 설명해주세요.
        5. 이전 대화 내용을 참고하여 중복 추천을 피해주세요.
        6. 일정 관련 질문에는 현재 여행 계획을 참고하여 답변해주세요.
        """

        # GPT에 전달할 메시지 구성
        messages = [
            {"role": "system", "content": system_content},
            *self.chat_history,  # 이전 대화 기록 포함
            {"role": "user", "content": question}
        ]
        
        # GPT 응답 생성
        response = await self.llm.agenerate([messages])
        answer = response.generations[0][0].text.strip()
        
        # 대화 기록 저장
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        # 대화 기록 길이 제한 (최근 6턴만 유지)
        if len(self.chat_history) > self.max_turns * 2:
            self.chat_history = self.chat_history[-self.max_turns * 2:]
        
        return answer

    def load_travel_plan_from_file(self, file_path: str = "app/travel/data/test.tmp") -> dict:
        """tmp 파일에서 여행 플랜 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # "=== 최종 여행 계획 ===" 이후의 JSON 부분만 파싱
                json_str = content.split("=== 최종 여행 계획 ===")[-1].strip()
                plan_data = json.loads(json_str)
                
                # result 키에서 첫 번째 장소의 주소에서 목적지 정보 추출
                if 'result' in plan_data and plan_data['result'].get('Day 1'):
                    first_place = plan_data['result']['Day 1'][0]['place1']
                    address = first_place['address']  # "예를 들어, 경상북도 경주시 불국로 385 불국사"
                    # 주소에서 도/시 정보 추출
                    address_parts = address.split()
                    # 목적지 정보 추가
                    plan_data['destination'] = address_parts[1].replace('시', '')  # "경주시" -> "경주"
                    plan_data['detail_destination'] = address_parts[1].replace('시', '')  # "경주시" -> "경주"
                
                return plan_data
        except Exception as e:
            print(f"여행 플랜 파일 로드 중 오류 발생: {e}")
            return {}

    def _load_csv_data(self, category: str) -> dict:
        """카테고리별 CSV 파일들을 지역별로 로드하고 통합"""
        data_by_region = {}  # 지역별 데이터를 저장할 딕셔너리
        category_path = self.data_path / category  # 카테고리별 경로 설정
        
        try:
            for csv_file in category_path.glob("*.csv"):  # 모든 CSV 파일 순회
                try:
                    # ANSI(cp949) 인코딩으로 먼저 시도
                    df = pd.read_csv(csv_file, encoding='cp949')
                except UnicodeDecodeError:
                    try:
                        # 실패하면 utf-8 시도
                        df = pd.read_csv(csv_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        # 마지막으로 euc-kr 시도
                        df = pd.read_csv(csv_file, encoding='euc-kr')
                
                # 파일명에서 지역명 추출 (예: "경주_food.csv" -> "경주")
                file_name = csv_file.stem
                region = file_name.split('_')[0]
                
                # 지역별로 데이터 통합
                if region in data_by_region:
                    data_by_region[region] = pd.concat([data_by_region[region], df], ignore_index=True)
                else:
                    data_by_region[region] = df
                
            return data_by_region
            
        except Exception as e:
            print(f"{category} 데이터 로드 중 오류 발생: {e}")
            return {}

    def search_local_data(self, query: str, category: str = None) -> str:
        """CSV 데이터에서 정보 검색 - 지역별 필터링"""
        results = []
        
        try:
            if not self.destination:
                return "목적지 정보가 없습니다."
            
            # 디버그 정보 출력
            print(f"\n=== CSV 데이터 검색 디버그 ===")
            print(f"목적지: {self.destination}")
            print(f"food_data keys: {self.food_data.keys()}")
            print(f"검색 조건: {category == 'food' or ('맛집' in query or '식당' in query or '점심' in query or '저녁' in query)}")
            
            # 음식점 검색 로직
            if category == 'food' or ("맛집" in query or "식당" in query or "점심" in query or "저녁" in query):
                food_key = f"{self.destination}맛집"
                print(f"찾는 food_key: {food_key}")
                if food_key in self.food_data:
                    print(f"food_data[{food_key}] 데이터 shape: {self.food_data[food_key].shape}")
                    region_food_data = self.food_data[food_key]
                    # 검색어와 관련된 음식점 찾기
                    food_matches = region_food_data[
                        region_food_data.apply(lambda x: x.astype(str).str.contains(query, case=False).any(), axis=1)
                    ]
                    print(f"매칭된 음식점 수: {len(food_matches) if not food_matches.empty else 0}")
                    if not food_matches.empty:
                        results.extend(self._format_food_results(food_matches[:3]))
                else:
                    print(f"{food_key}를 찾을 수 없음")
            
            # 관광지 검색 로직
            if category == 'travel' or category is None:
                if self.destination in self.travel_data:
                    region_travel_data = self.travel_data[self.destination]
                    # 검색어와 관련된 관광지 찾기
                    travel_matches = region_travel_data[
                        region_travel_data.apply(lambda x: x.astype(str).str.contains(query, case=False).any(), axis=1)
                    ]
                    if not travel_matches.empty:
                        results.extend(self._format_travel_results(travel_matches[:3]))
            
            # 검색 결과가 없는 경우
            if not results:
                return f"⚠️ {self.destination}의 {query}에 대한 정보를 찾을 수 없습니다."
            
            return "\n\n".join(results)
            
        except Exception as e:
            print(f"데이터 검색 중 오류 발생: {e}")
            return "데이터 검색 중 오류가 발생했습니다."
        
    def _format_festival_results(self, df: pd.DataFrame) -> list:
        """축제 데이터 포맷팅"""
        results = []
        for _, row in df.iterrows():  # 데이터프레임의 각 행을 순회
            info = f"🎉 축제 정보:\n"
            for col in df.columns:  # 각 열(컬럼)을 순회
                if pd.notna(row[col]):  # null이 아닌 값만 포함
                    info += f"{col}: {row[col]}\n"
            results.append(info)
        return results

    def _format_travel_results(self, df: pd.DataFrame) -> list:
        """관광 데이터 포맷팅"""
        results = []
        for _, row in df.iterrows():  # 데이터프레임의 각 행을 순회
            info = f"🏛 관광지 정보:\n"
            for col in df.columns:  # 각 열(컬럼)을 순회
                if pd.notna(row[col]):  # null이 아닌 값만 포함
                    info += f"{col}: {row[col]}\n"
            results.append(info)
        return results

    def _format_food_results(self, df: pd.DataFrame) -> list:
        """음식점 데이터 포맷팅"""
        results = []
        for _, row in df.iterrows():  # 데이터프레임의 각 행을 순회
            info = f"🍽 맛집 정보:\n"
            for col in df.columns:  # 각 열(컬럼)을 순회
                if pd.notna(row[col]):  # null이 아닌 값만 포함
                    info += f"{col}: {row[col]}\n"
            results.append(info)
        return results

    def _format_chat_history(self) -> str:
        """대화 기록 포맷팅"""
        if not self.chat_history:
            return "이전 대화 없음"
        
        formatted_history = []
        for msg in self.chat_history:  # 대화 기록의 각 메시지를 순회
            role = "사용자" if msg["role"] == "user" else "챗봇"  # 역할에 따라 표시 변경
            formatted_history.append(f"{role}: {msg['content']}")  # "역할: 내용" 형식으로 포맷팅
        
        return "\n".join(formatted_history)  # 각 메시지를 줄바꿈으로 구분하여 반환

    def find_nearby_places(self, base_address: str, places_df: pd.DataFrame, radius_km: float = 2.0) -> pd.DataFrame:
        """기준 주소 근처의 장소들을 찾는 함수"""
        # Nominatim 지오코더 초기화 (OpenStreetMap 기반 위치 검색 서비스)
        geolocator = Nominatim(user_agent="my_agent")
        
        # 기준 주소의 좌표(위도, 경도) 얻기
        base_location = geolocator.geocode(base_address)
        if not base_location:
            return places_df  # 좌표를 찾을 수 없으면 원본 ��이터프레임 반환
        
        # 기준 위치의 좌표
        base_coords = (base_location.latitude, base_location.longitude)
        
        # 각 장소의 거리 계산하는 함수
        def calculate_distance(address):
            try:
                # 주소로부터 좌표 얻기
                location = geolocator.geocode(address)
                if location:
                    coords = (location.latitude, location.longitude)
                    # geodesic: 두 지점 간의 최단 거리 계산 (km 단위)
                    return geodesic(base_coords, coords).kilometers
                return float('inf')  # 좌표를 찾을 수 없으면 무한대 거리 반환
            except:
                return float('inf')  # 에러 발생시 무한대 거리 반환
        
        # 각 장소의 거리 계산하여 새로운 컬럼 추가
        places_df['distance'] = places_df['주소'].apply(calculate_distance)
        
        # 지정된 반경(radius_km) 내의 장소들만 필터링하고 거리순으로 정렬
        nearby_places = places_df[places_df['distance'] <= radius_km].sort_values('distance')
        
        return nearby_places