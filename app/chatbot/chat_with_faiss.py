# chat_with_faiss.py (수정됨)
import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


# .env 파일 로드
env_path = os.path.join(os.path.dirname(__file__), 'env')
load_dotenv(dotenv_path=env_path)

# OpenAI 클라이언트 설정
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # .env 파일에서 API 키 관리
)

# SentenceTransformer 모델 로드 (jhgan/ko-sroberta-multitask 사용)
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 코사인 유사도 FAISS 인덱스 및 메타데이터 파일 경로 설정
base_dir = os.path.dirname(os.path.realpath(__file__))
index_file = os.path.join(base_dir, "../FAISS/Index/jhgan_cosine_index.bin")
metadata_file = os.path.join(base_dir, "../FAISS/Metadata/jhgan_metadata.pkl")

# FAISS 인덱스 및 메타데이터 로드
try:
    index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    print("코사인 유사도 기반 FAISS 인덱스 및 메타데이터가 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"FAISS 인덱스 또는 메타데이터 로드 실패: {e}")
    exit()

# LangChain 메모리 구성
memory = ConversationBufferMemory(memory_key="chat_history")

# PromptTemplate 정의
prompt_template = """사용자의 질문과 관련된 정보를 바탕으로 간단하고 정확한 답변을 제공해줘.

질문: {input}
추가 정보: {context}
이전 대화 히스토리: {chat_history}
답변:"""

prompt = PromptTemplate(input_variables=["input", "context", "chat_history"], template=prompt_template)

# 코사인 방식으로 RAG 응답 생성

def generate_rag_answer(input_text: str, context: str) -> str:
    prompt_text = prompt.format(input=input_text, context=context, chat_history=memory.buffer)
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_text}],
            model="gpt-3.5-turbo",
            max_tokens=300,
            temperature=0.3,
            top_p=0.95,
            n=1
        )
        response_text = response.choices[0].message.content.strip()
        memory.save_context({"input": input_text}, {"output": response_text})  # 메모리에 저장
    except Exception as e:
        print(f"Error during OpenAI API call (RAG): {e}")
        response_text = "죄송합니다, RAG 방식으로 응답을 생성하지 못했습니다."

    return response_text

# 코사인 유사도 기반 FAISS 인덱스에서 가장 유사한 문서 검색 함수
def search_faiss_index(query: str, top_k: int = 1, similarity_threshold: float = 0.5) -> dict:
    # 사용자 질문의 임베딩 생성
    query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)
    distances, indices_found = index.search(query_embedding, top_k)

    # 가장 유사한 문서의 텍스트와 거리 정보 가져오기
    best_result = {}
    if distances[0][0] < similarity_threshold:  # 유사도가 기준 거리 이상인 경우에만 반환
        if indices_found[0][0] < len(metadata):
            idx = indices_found[0][0]
            best_result = {
                "text_segment": metadata[idx]["text_segment"],
                "original_id": metadata[idx]["primary_id"],
                "segment_id": metadata[idx]["segment_id"],
                "distance": distances[0][0]
            }

    return best_result

# 챗봇 함수 정의
def process_chat(input_text: str) -> str:
    print(f"process_chat 함수가 호출되었습니다. 입력: {input_text}")

    # 유사한 문서 검색 및 메타데이터 추출
    best_result = search_faiss_index(input_text)
    if not best_result:
        # 유사한 문서가 없으면 기본 LLM 응답 생성 (대화 히스토리를 포함하여 응답)
        prompt_text = prompt.format(input=input_text, context="", chat_history=memory.buffer)
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_text}],
                model="gpt-3.5-turbo",
                max_tokens=300,
                temperature=0.3,
                top_p=0.95,
                n=1
            )
            response_text = response.choices[0].message.content.strip()
            memory.save_context({"input": input_text}, {"output": response_text})  # 메모리에 저장
        except Exception as e:
            print(f"Error during OpenAI API call (LLM): {e}")
            response_text = "죄송합니다, 응답을 생성하지 못했습니다."
        print(f"유사한 문서가 없습니다: {response_text}")
        return response_text
    
    context = best_result.get("text_segment", "")
    print(f"유사한 문서 검색 결과: {context[:100]}...")

    # RAG 방식으로 응답 생성
    rag_answer = generate_rag_answer(input_text, context)
    print(f"생성된 RAG 응답: {rag_answer}")
    
    # 현재 대화 히스토리 출력 (로그에 출력)
    print("=== 현재 대화 히스토리 ===")
    for entry in memory.buffer:
        # entry가 dict인지 확인 후 출력
        if isinstance(entry, dict) and "input" in entry and "output" in entry:
            print(f"User: {entry['input']}")
            print(f"AI: {entry['output']}")
        else:
            print(f"예상하지 못한 형식의 히스토리 항목: {entry}")
    print("=======================")
    
    # FastAPI 응답에 히스토리 포함
    history_output = "\n".join([f"User: {entry.get('input', 'N/A')} | AI: {entry.get('output', 'N/A')}" for entry in memory.buffer if isinstance(entry, dict)])
    
    # 결과 출력
    output = f"[코사인 방식 RAG 응답]\n{rag_answer}\n"
    output += f"(가장 유사한 문서 - 원본 ID: {best_result.get('original_id', 'N/A')}, 세그먼트 ID: {best_result.get('segment_id', 'N/A')}, 거리: {best_result.get('distance', 'N/A')})\n"
    output += f"컨텐츠: {context[:150]}...\n\n"
    output += f"=== 현재 대화 히스토리 ===\n{history_output}\n=======================\n"
    print("process_chat 결과 출력 완료")
    return output

# 디버깅 용도 - 사용자 입력과 모델 응답 출력
if __name__ == "__main__":
    user_input = "경복궁이 어떻게 만들어졌어?"
    print("디버깅 모드에서 사용자 입력 처리 중")
    print("Input from User:", user_input)
    print("\nResponses:\n", process_chat(user_input))
