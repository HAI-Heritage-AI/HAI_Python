#app/chatbot/chat.py
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.prompts import PromptTemplate

# .env 파일 로드
load_dotenv()

# OpenAI 클라이언트 설정
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # .env 파일에서 API 키 관리
)

# LangChain의 PromptTemplate 사용 - 간결한 응답 제공 유도
prompt_template = """사용자의 질문에 대해 간단하고 정확한 답변을 제공해줘.

질문: {input}
답변:"""

prompt = PromptTemplate(input_variables=["input"], template=prompt_template)

# 챗봇 함수 정의
def process_chat(input_text: str) -> str:
    # PromptTemplate를 사용하여 실제 프롬프트 문자열 생성
    prompt_text = prompt.format(input=input_text)
    
    try:
        # OpenAI GPT-3.5 API를 사용하여 응답 생성 (최신 인터페이스 사용)
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt_text}
            ],
            model="gpt-3.5-turbo",
            max_tokens=150,
            temperature=0.3,
            top_p=0.95,
            n=1
        )
        
        # 응답 텍스트 추출
        response_text = response.choices[0].message.content.strip()
    except Exception as e:
        # 예외 발생 시 디버그 메시지 출력 및 기본 응답 설정
        print(f"Error during OpenAI API call: {e}")
        response_text = "죄송합니다, 적절한 응답을 생성하지 못했습니다. 다시 한번 말씀해 주세요."

    # 응답의 길이 제약 추가
    if len(response_text) > 150:
        response_text = response_text[:150] + "..."

    return response_text

# 디버깅 용도 - 사용자 입력과 모델 응답 출력
if __name__ == "__main__":
    user_input = "FastAPI란 무엇인가요?"
    print("Input from User:", user_input)
    print("Response from GPT-3.5:", process_chat(user_input))