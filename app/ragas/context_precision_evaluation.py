import os
import time
import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import evaluate
from ragas.metrics._context_precision import ContextPrecision
from dotenv import load_dotenv
import pandas as pd

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

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
    query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            results.append({
                "text_segment": metadata[idx]["text_segment"],
                "distance": float(distances[0][i])
            })
    return results

# 데이터셋 로드
with open('national_heritage_qa_dataset_converted.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터 변환 및 검색 결과 저장
retrieval_results = []
samples = []

for item in data:
    query = item["question"]
    ground_truth = item["ground_truth"]

    retrieved_documents = search_faiss_index(query, top_k=5)
    retrieval_results.append({
        "query": query,
        "retrieved_documents": retrieved_documents,
        "ground_truth": ground_truth
    })
    samples.append(
        SingleTurnSample(
            user_input=query,
            retrieved_contexts=[doc["text_segment"] for doc in retrieved_documents],
            reference=ground_truth,
        )
    )

# 검색 결과 저장
output_file = "retrieval_results_ContextPrecision.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(retrieval_results, f, ensure_ascii=False, indent=4)
print(f"검색 결과가 {output_file}에 저장되었습니다.")

# RAGAS 데이터셋 생성
dataset = EvaluationDataset(samples=samples)

# 검색 성능 평가 메트릭 정의
metrics = [ContextPrecision()]

# 평가 실행
valid_samples = []
failed_samples = []

try:
    for i, sample in enumerate(dataset.samples):
        try:
            print(f"샘플 {i + 1}/{len(dataset.samples)} 평가 중...")
            result = evaluate(EvaluationDataset(samples=[sample]), metrics=metrics, show_progress=False)
            print(f"평가 결과: {result}")
            valid_samples.append(sample)  # 성공한 샘플 추가
        except Exception as e:
            print(f"샘플 {i + 1} 평가 실패: {e}")
            failed_samples.append({"sample": sample, "error": str(e)})  # 실패 샘플 저장

    # 실패한 샘플 저장
    if failed_samples:
        failed_output_file = "failed_samples.json"
        with open(failed_output_file, "w", encoding="utf-8") as f:
            json.dump(failed_samples, f, ensure_ascii=False, indent=4)
        print(f"실패한 샘플이 {failed_output_file}에 저장되었습니다.")

    # 성공한 샘플로 최종 평가
    if valid_samples:
        valid_dataset = EvaluationDataset(samples=valid_samples)
        final_results = evaluate(dataset=valid_dataset, metrics=metrics, show_progress=True)
        print("최종 평가 결과:")
        print(final_results)

        # pandas DataFrame으로 변환
        df = final_results.to_pandas()

        # JSON 저장
        json_output_file = "final_evaluation_results.json"
        df.to_json(json_output_file, orient="records", force_ascii=False, indent=4)
        print(f"최종 평가 결과가 {json_output_file}에 저장되었습니다.")

        # CSV 저장
        csv_output_file = "final_evaluation_results.csv"
        df.to_csv(csv_output_file, index=False, encoding="utf-8-sig")
        print(f"최종 평가 결과가 {csv_output_file}에 저장되었습니다.")
    else:
        print("평가에 성공한 샘플이 없습니다.")

except Exception as e:
    print(f"평가 중 알 수 없는 오류 발생: {e}")
