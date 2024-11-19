from langchain_openai import ChatOpenAI
from typing import Optional
import os
import requests
from dotenv import load_dotenv
from app.plan_agent import plan_travel, calculate_trip_days  # 추가
import json

load_dotenv()

class TravelChatAgent:
    def __init__(self):
        self.current_travel_plan = None
        self.destination = None
        self.travel_style = None
        self.user_info = None  # 사용자 정보 추가
        
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        self.chat_history = []
        self.max_turns = 6
        
        # 네이버 API 설정
        self.naver_headers = {
            "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
            "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")
        }

    def set_user_info(self, user_info: dict):
        """여행자 정보 설정"""
        self.user_info = user_info
        self.destination = user_info.get('destination')
        self.travel_style = user_info.get('style')
        self.current_travel_plan = user_info  # 여행 계획을 최신으로 설정

        # 여행 계획 생성 (옵션: 처음 생성이 아닐 때는 업데이트)
        if not self.current_travel_plan:
            try:
                result = plan_travel(self.user_info)
                if result:
                    self.current_travel_plan = json.loads(result)
            except Exception as e:
                print(f"여행 계획 생성 중 오류 발생: {e}")

    def _parse_travel_plan(self, context: str) -> dict:
        """여행 플랜에서 주요 정보 추출"""
        plan_info = {
            'places': [],  # 계획된 장소들
            'schedule': {} # 일정별 정보
        }
        
        try:
            # Day 1, Day 2 등으로 시작하는 일정 파싱
            days = context.split('[Day')
            for day in days[1:]:  # 첫 번째는 빈 문자열이므로 제외
                day_info = []
                lines = day.split('\n')
                current_day = lines[0].strip().rstrip(']')
                
                for line in lines:
                    if '주소:' in line:
                        place = {
                            'name': lines[lines.index(line)-1].split(':')[-1].strip(),
                            'address': line.split('주소:')[-1].strip(),
                        }
                        day_info.append(place)
                        plan_info['places'].append(place)
                
                plan_info['schedule'][current_day] = day_info
                
        except Exception as e:
            print(f"여행 플랜 파싱 중 오류 발생: {e}")
        
        return plan_info

    def search_naver_blog(self, query: str) -> str:
        """네이버 블로그 검색 - 지역 필터링 추가"""
        if not self.destination:
            print("Warning: destination이 설정되지 않았습니다!")
            return "여행 목적지 정보가 없습니다."

        # URL 정의 추가
        url = "https://openapi.naver.com/v1/search/blog"
        
        print(f"현재 설정된 destination: {self.destination}")
        
        # 검색어에 destination을 앞에 명확하게 포함
        search_query = f"{self.destination} {query}"
        params = {
            "query": search_query,
            "display": 10,
            "sort": "sim"
        }
        
        print(f"\n=== 네이버 블로그 검색 요청 ===")
        print(f"검색어: {search_query}")
        
        response = requests.get(url, headers=self.naver_headers, params=params)
        print(f"응답 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            items = response.json().get('items', [])
            filtered_items = []
            
            print(f"\n검색된 블로그 글 목록:")
            for item in items:
                print(f"\n제목: {item['title'].replace('<b>', '').replace('</b>', '')}")
                print(f"링크: {item['link']}")
                
                # 제목이나 내용에 destination이 포함된 결과만 필터링
                if self.destination in item['title'] or self.destination in item['description']:
                    filtered_items.append(item)
            
            if not filtered_items:
                return f"{self.destination}의 관련 정보를 찾을 수 없습니다."
            
            # 가장 관련성 높은 첫 번째 결과 선택
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
        """네이버 지역 검색 - 지역 필터링 추가"""
        url = "https://openapi.naver.com/v1/search/local"
        params = {
            "query": f"{self.destination} {query}",
            "display": 5,
            "sort": "random"
        }
        
        response = requests.get(url, headers=self.naver_headers, params=params)
        
        if response.status_code == 200:
            items = response.json().get('items', [])
            filtered_items = []
            
            for item in items:
                # 주소에 destination이 포함된 결과만 필터링
                if self.destination in item['address']:
                    filtered_items.append(item)
            
            if not filtered_items:
                return f"{self.destination}의 관련 장소를 찾을 수 없습니다."
            
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
        """챗봇 답변 생성 - 여행 플랜 고려"""
        if not self.destination:
            return "죄송합니다. 여행 목적지 정보가 없습니다."
            
        # 채팅 기록 관리
        if len(self.chat_history) > self.max_turns * 2:
            self.chat_history = self.chat_history[-self.max_turns * 2:]

        # 검색 수행
        blog_results = self.search_naver_blog(question)
        local_results = self.search_naver_local(question)

        # GPT 프롬프트 구성
        system_content = f"""당신은 {self.destination} 지역 전문 여행 챗봇입니다.
        현재 계획된 여행 정보:
        - 목적지: {self.destination}
        - 여행 스타일: {self.travel_style if self.travel_style else '정보 없음'}
        
        중요: 반드시 {self.destination} 지역의 정보만 추천해주세요.
        다른 도시의 정보는 추천하지 마세요.
        
        여행 계획: {context if context else json.dumps(self.current_travel_plan)}
        """

        messages = [
            {"role": "system", "content": system_content}
        ]
        
        # 이전 대화 기록 추가
        messages.extend(self.chat_history)
        
        # 현재 질문 관련 정보 추가
        messages.append({"role": "user", "content": f"""
            질문: {question}
            
            네이버 블로그 검색 결과:
            {blog_results}
            
            네이버 지역 검색 결과:
            {local_results}
        """})
        
        # GPT 응답 생성
        response = await self.llm.agenerate([messages])
        answer = response.generations[0][0].text.strip()
        
        # 대화 기록 저장
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return answer
