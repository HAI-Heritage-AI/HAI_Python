from langchain_openai import ChatOpenAI
from typing import Optional
import os
import requests
from dotenv import load_dotenv

load_dotenv()

class TravelChatAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        self.chat_history = []
        self.max_turns = 6  # 6턴의 대화 유지 (사용자 6번, 봇 6번)
        
        # 네이버 API 설정
        self.naver_headers = {
            "X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"),
            "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")
        }
    
    def search_naver_blog(self, query: str) -> str:
        """네이버 블로그 검색"""
        url = "https://openapi.naver.com/v1/search/blog"
        params = {
            "query": query,
            "display": 5,
            "sort": "date"
        }
        
        print(f"\n=== 네이버 블로그 검색 요청 ===")
        print(f"검색어: {query}")
        
        response = requests.get(url, headers=self.naver_headers, params=params)
        
        print(f"응답 상태 코드: {response.status_code}")
        if response.status_code == 200:
            results = response.json().get("items", [])
            print(f"검색 결과 수: {len(results)}")
            formatted_results = []
            
            for item in results:
                title = item.get("title").replace("<b>", "").replace("</b>", "")
                description = item.get("description").replace("<b>", "").replace("</b>", "")
                formatted_results.append(f"제목: {title}\n내용: {description}\n")
                print(f"\n블로그 글: {title}")
            
            return "\n".join(formatted_results)
        return "블로그 검색 결과를 가져오는데 실패했습니다."

    def search_naver_local(self, query: str) -> str:
        """네이버 지역 검색"""
        url = "https://openapi.naver.com/v1/search/local"
        params = {
            "query": query,
            "display": 5
        }
        
        print(f"\n=== 네이버 지역 검색 요청 ===")
        print(f"검색어: {query}")
        
        response = requests.get(url, headers=self.naver_headers, params=params)
        
        print(f"응답 상태 코드: {response.status_code}")
        if response.status_code == 200:
            results = response.json().get("items", [])
            print(f"검색 결과 수: {len(results)}")
            formatted_results = []
            
            for item in results:
                title = item.get("title").replace("<b>", "").replace("</b>", "")
                address = item.get("address")
                roadAddress = item.get("roadAddress")
                formatted_results.append(f"이름: {title}\n주소: {address}\n도로명: {roadAddress}\n")
                print(f"\n장소: {title}\n주소: {address}")
            
            return "\n".join(formatted_results)
        return "지역 검색 결과를 가져오는데 실패했습니다."
        
    async def get_answer(self, question: str, context: str) -> str:
        # 채팅 기록이 너무 길어지면 오래된 대화 삭제
        if len(self.chat_history) > self.max_turns * 2:
            self.chat_history = self.chat_history[-self.max_turns * 2:]
        
        # 네이버 검색 수행
        blog_results = self.search_naver_blog(question)
        local_results = self.search_naver_local(question)
        
        # 메시지 구성 - 중요한 시스템 프롬프트 유지
        messages = [
            {"role": "system", "content": """당신은 여행 전문 챗봇입니다. 
            사용자의 여행 계획을 이해하고, 그에 맞춰 도움을 주세요.
            네이버 검색 결과와 기존 여행 계획을 참고하여 
            맥락에 맞는 상세한 답변을 제공해주세요.
            
            특히 다음 사항을 고려해주세요:
            1. 기존 여행 계획의 시간과 장소를 고려한 답변
            2. 이동 거리와 소요 시간을 고려한 현실적인 제안
        """}
        ]
        
        # 이전 대화 기록 추가
        messages.extend(self.chat_history)
        
        # 현재 질문 추가
        messages.append({"role": "user", "content": f"""
            현재 여행 계획:
            {context}
            
            사용자 질문: {question}
            
            네이버 블로그 검색 결과:
            {blog_results}
            
            네이버 지역 검색 결과:
            {local_results}
        """})
        
        # ChatGPT API 호출
        response = await self.llm.agenerate([messages])
        answer = response.generations[0][0].text
        
        # 대화 기록 저장
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        return answer