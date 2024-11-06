import os
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.preprocessing import normalize

# 1. FAISS 인덱스와 메타데이터 파일 경로 설정
base_dir = os.path.dirname(os.path.realpath(__file__))  # 현재 파일의 경로를 가져옵니다.
index_file = os.path.join(base_dir, "../FAISS/Index/faiss_index_1000_cosine.bin")
metadata_file = os.path.join(base_dir, "../FAISS/Metadata/faiss_metadata_1000_cosine.pkl")

# 2. FAISS 인덱스 불러오기 (코사인 유사도를 사용하기 위해 IndexFlatIP로 생성)
try:
    # 기존 인덱스를 파일 경로에서 불러옵니다.
    index = faiss.read_index(index_file)
    print(f"FAISS 인덱스 '{index_file}'을(를) 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"FAISS 인덱스를 불러오는 데 실패했습니다: {e}")
    exit()

# 3. 메타데이터 불러오기
try:
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    print(f"메타데이터 '{metadata_file}'을(를) 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"메타데이터를 불러오는 데 실패했습니다: {e}")
    exit()

# 4. 사용자의 임시 질문을 입력받아 임베딩
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
user_question = "경복궁이 어떻게 만들어졌어?"  # 사용자의 임시 질문
embedding = model.encode(user_question).astype('float32').reshape(1, -1)
embedding = normalize(embedding, norm='l2')  # 벡터를 L2 정규화하여 코사인 유사도에 맞춤

# 5. FAISS 인덱스에서 유사한 벡터 검색
D, I = index.search(embedding, k=5)  # 가장 유사한 5개 검색 (내적 사용)

print(f"가장 유사한 벡터들의 인덱스: {I}")
print(f"각 유사한 벡터와의 유사도: {D}")

# 6. 검색 결과와 메타데이터 매핑
for idx in I[0]:
    if idx < len(metadata):  # 메타데이터의 인덱스 범위 확인
        data = metadata[idx]
        # 메타데이터가 딕셔너리인지 확인
        if isinstance(data, dict):
            print(f"원본 데이터 ID: {data['original_id']}, 세그먼트 번호: {data['segment_id']}, 텍스트 세그먼트: {data['text_segment']}")
        else:
            print(f"인덱스 {idx}의 메타데이터 형식이 잘못되었습니다. 데이터: {data}")
    else:
        print(f"인덱스 {idx}는 메타데이터 범위를 벗어났습니다.")
