# chat_with_hybrid.py
import os 
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from eunjeon import Mecab
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import pandas as pd
import time

load_dotenv()

# OpenAI 클라이언트 설정
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # .env 파일에서 API 키 관리
)

# SentenceTransformer 모델 로드
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# Mecab 형태소 분석기 초기화
mecab = Mecab()

# FAISS 인덱스 및 메타데이터 파일 경로 설정
base_dir = os.path.dirname(os.path.realpath(__file__))
index_file = os.path.join(base_dir, "../FAISS/Index/jhgan_cosine_index.bin")
metadata_file = os.path.join(base_dir, "../FAISS/Metadata/jhgan_metadata.pkl")
bm25_index_file = os.path.join(base_dir, "../FAISS/Metadata/bm25_index.pkl")

# FAISS 인덱스 및 메타데이터 로드
try:
    index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    print("FAISS 인덱스 및 메타데이터 로드 성공")
except Exception as e:
    print(f"FAISS 인덱스 또는 메타데이터 로드 실패: {e}")
    exit()

# 문서 리스트 생성
documents = [entry["내용"] for entry in metadata]

# BM25 인덱스 로드 또는 생성
if os.path.exists(bm25_index_file):
    with open(bm25_index_file, "rb") as f:
        bm25 = pickle.load(f)
    print("BM25 인덱스 로드 성공")
else:
    print("BM25 인덱스 생성 중...")
    tokenized_documents = [[word for word, pos in mecab.pos(doc) if pos in ['NNP', 'NNG', 'NP', 'VV', 'VA', 'VCP', 'VCN', 'VSV', 'MAG', 'MAJ']] for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)
    with open(bm25_index_file, "wb") as f:
        pickle.dump(bm25, f)
    print("BM25 인덱스 생성 및 저장 성공")

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
'''

## 임무 
''' 
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

# 추가 정보
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
            max_tokens=500,
            temperature=0.3,
            top_p=0.95,
            n=1
        )
        response_text = response.choices[0].message.content.strip()
        memory.save_context({"input": input_text}, {"output": response_text})
    except Exception as e:
        print(f"Error during OpenAI API call (RAG): {e}")
        response_text = "죄송합니다, RAG 방식으로 응답을 생성하지 못했습니다."
    return response_text

# 하이브리드 서치 구현
def hybrid_search(query: str, k: int = 3, alpha: float = 0.5, normalization_method: str = "min_max") -> dict:
    query_tokens = [word for word, pos in mecab.pos(query) if pos in ['NNP', 'NNG', 'NP', 'VV', 'VA', 'VCP', 'VCN', 'VSV', 'MAG', 'MAJ']]
    bm25_scores = bm25.get_scores(query_tokens)
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)
    faiss_scores, faiss_indices = index.search(query_embedding, len(documents))
    faiss_scores = faiss_scores[0]
    
    # 점수 정규화
    if normalization_method == "min_max":
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        faiss_scores = (faiss_scores - np.min(faiss_scores)) / (np.max(faiss_scores) - np.min(faiss_scores))
    elif normalization_method == "z_score":
        bm25_scores = (bm25_scores - np.mean(bm25_scores)) / np.std(bm25_scores)
        faiss_scores = (faiss_scores - np.mean(faiss_scores)) / np.std(faiss_scores)
    elif normalization_method == "max":
        bm25_scores = bm25_scores / np.max(bm25_scores)
        faiss_scores = faiss_scores / np.max(faiss_scores)
    else:
        raise ValueError("지원하지 않는 정규화 방법입니다. 'min_max', 'z_score', 'max' 중 선택하세요.")

    final_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores
    sorted_indices = np.argsort(-final_scores)[:k]

    results = [metadata[i] for i in sorted_indices]
    return results

# 챗봇 함수 정의
def process_chat(input_text: str) -> str:
    start_time = time.time()  # 시작 시간 기록
    best_results = hybrid_search(input_text, k=3)
    context = "\n".join([result.get("내용", "") for result in best_results])
    
    # 하이브리드 서치 결과 출력
    print("하이브리드 서치 결과:")
    for idx, result in enumerate(best_results, 1):
        print(f"[{idx}] {result}")

    rag_answer = generate_rag_answer(input_text, context)
    end_time = time.time()  # 종료 시간 기록
    print(f"process_chat 함수 실행 시간: {end_time - start_time:.2f}초")  # 실행 시간 출력
    return rag_answer

# 디버깅 용도 - 사용자 입력과 모델 응답 출력
if __name__ == "__main__":
    user_input = "나무로 만들어진 문화유산"
    print("디버깅 모드에서 사용자 입력 처리 중")
    print("Input from User:", user_input)
    print("\nResponses:\n", process_chat(user_input))
