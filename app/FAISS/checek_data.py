import faiss
import numpy as np
import pickle

# 1. FAISS 인덱스 불러오기
index_file = "faiss_index.bin"
try:
    index = faiss.read_index(index_file)
    print(f"FAISS 인덱스 '{index_file}'을(를) 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"FAISS 인덱스를 불러오는 데 실패했습니다: {e}")
    exit()

# 2. 저장된 벡터의 개수 확인
total_vectors = index.ntotal
print(f"현재 FAISS 인덱스에 저장된 벡터의 개수: {total_vectors}")

# 3. 벡터 일부 확인 (저장된 벡터 중 첫 5개 출력)
if total_vectors > 0:
    num_vectors_to_check = total_vectors  # 확인할 벡터의 개수 (최대 5개)
    for i in range(num_vectors_to_check):
        vector = index.reconstruct(i)
        print(f"벡터 {i}: {vector[:10]}...")  # 첫 10개의 요소만 출력
else:
    print("FAISS 인덱스에 저장된 벡터가 없습니다.")

# 4. 메타데이터 불러오기
metadata_file = "faiss_metadata.pkl"
try:
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    print(f"메타데이터 파일 '{metadata_file}'을(를) 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"메타데이터 파일을 불러오는 데 실패했습니다: {e}")
    exit()

# 5. 메타데이터 확인 (저장된 메타데이터 중 첫 5개 출력)
if len(metadata) > 0:
    for i in range(num_vectors_to_check):
        print(f"벡터 {i}에 해당하는 원본 데이터 ID: {metadata[i]}")
else:
    print("메타데이터가 없습니다.")
