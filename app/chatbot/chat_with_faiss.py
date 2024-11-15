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

load_dotenv()

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
prompt_template = """
# 사용자의 페르소나
'''한국 문화유산에 대해 관심이 많은 사람이며, 문화유산에 대해 잘 모르는 사람.'''

# AI(화자)

## 페르소나
'''
수십 년 경력의 문화유산 해설사로서, 문화유산에 대해 궁금한 점이 많은 사람들에게 양질의 정보를 제공하고자 합니다.
답변은 300토큰 이내로 답변해야 합니다.
'''

## 임무 
''' 
답변은 300토큰 이내로 답변해야 합니다.
사용자의 질문에 대해서 양질의 정보를 제공하고, 자연스럽고 인간적인 방식으로 답변을 드리며, 편견에 의존하지 않고, 단계별로 사고해 대답합니다.
사용자의 질문이 모호하거나 정보가 부족할 경우, 충분한 정보를 얻기 위해 역질문을 할 수 있습니다. 
답변 이후에 추가로 궁금한 점이 있는지 물어보아, 대화의 연속성을 유도합니다.
항상 이전 대화 히스토리를 고려하여 이전 대화와 연결성 있는 답변을 생성합니다.
좋은 답변을 할 경우 팁을 받을 수 있습니다.
목표를 달성하지 못할 경우, 벌금이 부과될 수 있습니다.
'''

## 사용자 참여 유도: 
- 맥락 관련 요청: 기존 대화 흐름을 고려해 자연스럽게 응답합니다.

# 사용자의 질문
'''
{input}
'''

# 추가 정보d
''' 
{context}
'''

# 이전 대화 히스토리
'''
{chat_history}
'''

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
def search_faiss_index(query: str, top_k: int = 5, similarity_threshold: float = 10) -> dict:
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
    # print(f"process_chat 함수가 호출되었습니다. 입력: {input_text}")

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
    # print(f"유사한 문서 검색 결과: {context[:100]}...")

    # RAG 방식으로 응답 생성
    rag_answer = generate_rag_answer(input_text, context)
    # print(f"생성된 RAG 응답: {rag_answer}")
    
    # FastAPI 응답에 히스토리 포함
    history_output = "\n".join([f"User: {entry.get('input', 'N/A')} | AI: {entry.get('output', 'N/A')}" for entry in memory.buffer if isinstance(entry, dict)])
    
    # 결과 출력
    output = f"{rag_answer}"
    output += f"(가장 유사한 문서 - 원본 ID: {best_result.get('original_id', 'N/A')}, 세그먼트 ID: {best_result.get('segment_id', 'N/A')}, 거리: {best_result.get('distance', 'N/A')})\n"
    # output += f"컨텐츠: {context[:150]}...\n\n"
    # output += f"=== 현재 대화 히스토리 ===\n{history_output}\n=======================\n"
    # print("process_chat 결과 출력 완료")
    return output

# 디버깅 용도 - 사용자 입력과 모델 응답 출력
if __name__ == "__main__":
    user_input = "현재 서울에 남아 있는 가장 오래된 목조 건물은 언제 완성되었나요?"
    print("디버깅 모드에서 사용자 입력 처리 중")
    print("Input from User:", user_input)
    print("\nResponses:\n", process_chat(user_input))
