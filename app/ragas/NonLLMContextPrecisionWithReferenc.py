import json
import csv
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import evaluate
from ragas.metrics._context_precision import NonLLMContextPrecisionWithReference

# FAISS 인덱스 및 메타데이터 파일 로드
index_file = "../FAISS/Index/jhgan_cosine_index.bin"
metadata_file = "../FAISS/Metadata/jhgan_metadata.pkl"

try:
    index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)
    print("FAISS 인덱스 및 메타데이터 로드 성공!")
except Exception as e:
    print(f"FAISS 인덱스 및 메타데이터 로드 실패: {e}")
    exit()

# SentenceTransformer 모델 로드
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 검색 함수
def search_faiss_index(query: str, top_k: int = 5):
    """
    FAISS 인덱스를 사용하여 코사인 유사도 기반 검색 수행.
    """
    query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append({
                "text_segment": metadata[idx]["text_segment"],
                "distance": float(distances[0][i])  # float32 -> float 변환
            })
    return results

# 데이터셋 로드
with open('national_heritage_qa_dataset_converted.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 검색 결과 저장용 리스트
retrieval_results = []

# 데이터셋 변환
samples = []
for item in data:
    query = item["question"]  # 사용자가 묻는 질문
    ground_truth = item["ground_truth"]  # 정답 (단일 문자열)

    # FAISS를 사용해 검색 수행
    retrieved_documents = search_faiss_index(query, top_k=5)

    # 검색 결과 저장
    retrieval_results.append({
        "query": query,
        "retrieved_documents": retrieved_documents,
        "ground_truth": ground_truth
    })

    # RAGAS SingleTurnSample 객체 생성
    samples.append(
        SingleTurnSample(
            user_input=query,  # 질문
            retrieved_contexts=[doc["text_segment"] for doc in retrieved_documents],  # 검색된 결과
            reference=ground_truth,  # 정답
            reference_contexts=[ground_truth],  # 참조 컨텍스트 추가
        )
    )

# 검색 결과를 JSON 파일로 저장
output_json_file = "retrieval_results_NonLLMContextPrecisionWithReferenc.json"
with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(retrieval_results, f, ensure_ascii=False, indent=4)
print(f"검색 결과가 {output_json_file}에 저장되었습니다.")

# RAGAS 데이터셋 생성
dataset = EvaluationDataset(samples=samples)

# 검색 성능 평가 메트릭 정의
metrics = [NonLLMContextPrecisionWithReference()]

# 평가 실행
try:
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=None,  # LLM을 사용하지 않음
        embeddings=None,  # 임베딩을 사용하지 않음
        show_progress=True
    )
    print("RAGAS 검색 성능 평가 결과:")
    print(results)

    # 평가 결과를 JSON 및 CSV로 저장
    output_results_json_file = "evaluation_results_NonLLMContextPrecisionWithReferenc.json"
    output_results_csv_file = "evaluation_results_NonLLMContextPrecisionWithReferenc.csv"

    # 결과를 직접 처리
    results_dict = {metric.name: results[metric.name] for metric in metrics}

    # JSON 저장
    with open(output_results_json_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    print(f"평가 결과가 {output_results_json_file}에 저장되었습니다.")

    # CSV 저장
    with open(output_results_csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Score"])  # 헤더 작성
        for metric, score in results_dict.items():
            writer.writerow([metric, score])
    print(f"평가 결과가 {output_results_csv_file}에 저장되었습니다.")
except Exception as e:
    print(f"평가 중 오류 발생: {e}")
