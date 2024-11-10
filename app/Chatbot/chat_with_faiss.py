import os
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_core.prompts import PromptTemplate

# .env 파일 로드
env_path = os.path.join(os.path.dirname(__file__), 'env')
load_dotenv(dotenv_path=env_path)

# OpenAI 클라이언트 설정
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # .env 파일에서 API 키 관리
)

# SentenceTransformer 모델 로드 (jhgan/ko-sroberta-multitask 사용)
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# FAISS 인덱스 및 메타데이터 파일 로드 경로 설정
base_dir = os.path.dirname(os.path.realpath(__file__))
index_files = {
    "dot_product": os.path.join(base_dir, "../FAISS/Index/jhgan_dotProduct_index.bin"),
    "cosine": os.path.join(base_dir, "../FAISS/Index/jhgan_cosine_index.bin"),
    "euclidean": os.path.join(base_dir, "../FAISS/Index/jhgan_euclidean_index.bin"),
}
metadata_file = os.path.join(base_dir, "../FAISS/Metadata/jhgan_metadata.pkl")

# FAISS 인덱스 및 메타데이터 로드
try:
    indices = {name: faiss.read_index(path) for name, path in index_files.items()}
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    print("FAISS 인덱스 및 메타데이터가 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"FAISS 인덱스 또는 메타데이터 로드 실패: {e}")
    exit()

# LangChain의 PromptTemplate 사용 - 간결한 응답 제공 유도
prompt_template = """사용자의 질문과 관련된 정보를 바탕으로 간단하고 정확한 답변을 제공해줘.

질문: {input}
추가 정보: {context}
답변:"""

prompt = PromptTemplate(input_variables=["input", "context"], template=prompt_template)

# FAISS를 이용한 관련 문서 검색 함수
def search_faiss_index(query: str, index, top_k: int = 1) -> dict:
    # 사용자 질문의 임베딩 생성
    query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)

    # FAISS 인덱스를 통해 유사한 문서 검색
    distances, indices_found = index.search(query_embedding, top_k)

    # 가장 유사한 문서의 텍스트와 거리 정보 가져오기
    best_result = {}
    if indices_found[0][0] < len(metadata):
        idx = indices_found[0][0]
        best_result = {
            "text_segment": metadata[idx]["text_segment"],
            "original_id": metadata[idx]["primary_id"],
            "segment_id": metadata[idx]["segment_id"],
            "distance": distances[0][0]
        }

    return best_result

# ChatGPT를 사용하여 RAG 방식으로 답변 생성
def generate_rag_answer(input_text: str, context: str) -> str:
    prompt_text = prompt.format(input=input_text, context=context)
    
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
    except Exception as e:
        print(f"Error during OpenAI API call (RAG): {e}")
        response_text = "죄송합니다, RAG 방식으로 응답을 생성하지 못했습니다."

    return response_text

# ChatGPT 단독 답변 생성
def generate_chatgpt_answer(input_text: str) -> str:
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": input_text}],
            model="gpt-3.5-turbo",
            max_tokens=300,
            temperature=0.3,
            top_p=0.95,
            n=1
        )
        response_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during OpenAI API call (ChatGPT only): {e}")
        response_text = "죄송합니다, ChatGPT만을 사용하여 응답을 생성하지 못했습니다."

    return response_text

# 챗봇 함수 정의
def process_chat(input_text: str) -> str:
    # 기본 LLM 답변 생성
    llm_answer = generate_chatgpt_answer(input_text)
    
    # 각 방식의 FAISS 인덱스 검색 및 RAG 응답 생성
    rag_answers = {}
    for method, index in indices.items():
        # 유사한 문서 검색 및 메타데이터 추출
        best_result = search_faiss_index(input_text, index)
        context = best_result["text_segment"]
        distance = best_result["distance"]
        
        # RAG 방식으로 응답 생성
        rag_answer = generate_rag_answer(input_text, context)
        
        # RAG 응답과 유사 문서 정보 저장
        rag_answers[method] = {
            "answer": rag_answer,
            "metadata": best_result
        }

    # 각 결과 출력
    output = "[기본 LLM 응답]\n" + llm_answer + "\n\n"
    for method, result in rag_answers.items():
        metadata = result["metadata"]
        output += f"[{method.capitalize()} 방식 RAG 응답]\n{result['answer']}\n"
        output += f"(가장 유사한 문서 - 원본 ID: {metadata['original_id']}, 세그먼트 ID: {metadata['segment_id']}, 거리: {metadata['distance']})\n"
        output += f"컨텐츠: {metadata['text_segment'][:150]}...\n\n"  # 일부 컨텐츠만 출력
    return output

# 디버깅 용도 - 사용자 입력과 모델 응답 출력
if __name__ == "__main__":
    user_input = "경복궁이 어떻게 만들어졌어?"
    print("Input from User:", user_input)
    print("\nResponses:\n", process_chat(user_input))
