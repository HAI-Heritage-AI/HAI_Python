import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.preprocessing import normalize

# 1. 경로 설정 및 임베딩 파일 로드 함수
base_dir = os.path.dirname(os.path.realpath(__file__))
embedding_paths = {
    "dotProduct": ("../FAISS/Index/jhgan_dotProduct_index_1000.bin", "../FAISS/Metadata/jhgan_metadata_1000.pkl"),
    "cosine": ("../FAISS/Index/jhgan_cosine_index_1000.bin", "../FAISS/Metadata/jhgan_metadata_1000.pkl"),
    "euclidean": ("../FAISS/Index/jhgan_euclidean_index_1000.bin", "../FAISS/Metadata/jhgan_metadata_1000.pkl")
}

# 2. SentenceTransformer 모델 로드 및 질문 설정
model = SentenceTransformer('jhgan/ko-sroberta-multitask')  # 모델을 jhgan/ko-sroberta-multitask로 변경
user_question = "경복궁이 어떻게 만들어졌어?"  # 사용자의 임시 질문
embedding = model.encode(user_question).astype('float32').reshape(1, -1)

# 3. 검색 함수 정의
def search_faiss_index(index_file, metadata_file, embedding, normalize_embedding=False):
    # FAISS 인덱스 불러오기
    try:
        index = faiss.read_index(index_file)
        print(f"FAISS 인덱스 '{index_file}'을(를) 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"FAISS 인덱스를 불러오는 데 실패했습니다: {e}")
        return

    # 메타데이터 불러오기
    try:
        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)
        print(f"메타데이터 '{metadata_file}'을(를) 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f"메타데이터를 불러오는 데 실패했습니다: {e}")
        return

    # 필요 시 임베딩 정규화 (코사인 유사도)
    if normalize_embedding:
        embedding = normalize(embedding, norm='l2')

    # 유사도 검색
    D, I = index.search(embedding, k=5)
    print(f"가장 유사한 벡터들의 인덱스: {I}")
    print(f"각 유사한 벡터와의 유사도: {D}")

    # 검색 결과와 메타데이터 매핑
    for idx in I[0]:
        if idx < len(metadata):
            data = metadata[idx]
            if isinstance(data, dict):
                print(f"원본 데이터 ID: {data['original_id']}, 세그먼트 번호: {data['segment_id']}, 텍스트 세그먼트: {data['text_segment']}")
            else:
                print(f"인덱스 {idx}의 메타데이터 형식이 잘못되었습니다. 데이터: {data}")
        else:
            print(f"인덱스 {idx}는 메타데이터 범위를 벗어났습니다.")

# 4. 각 방식에 따른 검색 실행
print("내적 방식으로 검색")
search_faiss_index(embedding_paths["dotProduct"][0], embedding_paths["dotProduct"][1], embedding)

print("\n코사인 유사도 방식으로 검색")
search_faiss_index(embedding_paths["cosine"][0], embedding_paths["cosine"][1], embedding, normalize_embedding=True)

print("\n유클리드 거리 방식으로 검색")
search_faiss_index(embedding_paths["euclidean"][0], embedding_paths["euclidean"][1], embedding)
