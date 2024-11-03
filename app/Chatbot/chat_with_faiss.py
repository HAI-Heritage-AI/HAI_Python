#app/chatbot/chat.py
import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

# .env 파일 로드
load_dotenv()

# OpenAI 클라이언트 설정
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # .env 파일에서 API 키 관리
)

# LangChain의 PromptTemplate 사용 - 간결한 응답 제공 유도
prompt_template = """사용자의 질문에 대해 간단하고 정확한 답변을 제공해줘. 참고 문서를 기반으로 응답을 생성해줘.

질문: {input}
참고 문서: {context}
답변:"""

prompt = PromptTemplate(input_variables=["input", "context"], template=prompt_template)

# FAISS 인덱스 및 메타데이터 불러오기
index_file = "faiss_index_1000.bin"
metadata_file = "faiss_metadata_1000.pkl"

try:
    index = faiss.read_index(index_file)
    print(f"FAISS 인덱스 '{index_file}'을(를) 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"FAISS 인덱스를 불러오는 데 실패했습니다: {e}")
    exit()

try:
    with open(metadata_file, "rb") as f:
        original_ids = pickle.load(f)
    print(f"메타데이터 파일 '{metadata_file}'을(를) 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"메타데이터 파일을 불러오는 데 실패했습니다: {e}")
    exit()

# 임베딩 모델 로드
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 챗봇 함수 정의 (RAG)
def process_chat(input_text: str) -> str:
    # 1. 사용자 질문 임베딩
    query_embedding = embedding_model.encode(input_text).astype('float32')
    
    # 2. FAISS를 사용하여 유사한 벡터 검색
    query_vector = query_embedding.reshape(1, -1)
    D, I = index.search(query_vector, k=5)  # 가장 유사한 5개 검색

    # 3. 검색 결과를 이용해 참고 문서 생성
    retrieved_texts = []
    for idx in I[0]:
        if idx < len(original_ids):
            retrieved_texts.append(original_ids[idx])
    retrieval_context = " ".join(retrieved_texts)  # 검색된 텍스트들을 하나로 합침

    # 4. PromptTemplate를 사용하여 실제 프롬프트 문자열 생성
    prompt_text = prompt.format(input=input_text, context=retrieval_context)

    # 5. OpenAI GPT-3.5 API를 사용하여 응답 생성 (최신 인터페이스 사용)
    try:
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
