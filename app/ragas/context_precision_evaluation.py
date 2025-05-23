import os
import json
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import evaluate
from ragas.metrics._context_precision import ContextPrecision
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from eunjeon import Mecab

# 환경 변수 로드
load_dotenv()

# 전역 변수 초기화
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
mecab = Mecab()

# FAISS 인덱스 및 메타데이터 로드
base_dir = os.path.dirname(os.path.realpath(__file__))
index_file = os.path.join(base_dir, "../FAISS/Index/jhgan_cosine_index.bin")
metadata_file = os.path.join(base_dir, "../FAISS/Metadata/jhgan_metadata.pkl")
bm25_index_file = os.path.join(base_dir, "../FAISS/Metadata/bm25_index.pkl")

with open(metadata_file, "rb") as f:
    metadata = pickle.load(f)
documents = [entry["내용"] for entry in metadata]

index = faiss.read_index(index_file)

if os.path.exists(bm25_index_file):
    with open(bm25_index_file, "rb") as f:
        bm25 = pickle.load(f)
else:
    tokenized_documents = [[word for word, pos in mecab.pos(doc) if pos in ['NNP', 'NNG', 'NP', 'VV', 'VA']] for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)

# 하이브리드 서치 함수
def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5, normalization_method: str = "min_max"):
    query_tokens = [word for word, pos in mecab.pos(query) if pos in ['NNP', 'NNG', 'NP', 'VV', 'VA']]
    bm25_scores = bm25.get_scores(query_tokens)

    query_embedding = embedding_model.encode(query).astype("float32").reshape(1, -1)
    faiss_distances, faiss_indices = index.search(query_embedding, len(metadata))
    faiss_scores = -faiss_distances[0]

    if normalization_method == "min_max":
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        faiss_scores = (faiss_scores - np.min(faiss_scores)) / (np.max(faiss_scores) - np.min(faiss_scores))

    final_scores = alpha * bm25_scores + (1 - alpha) * faiss_scores
    sorted_indices = np.argsort(-final_scores)[:top_k]

    results = [{"text_segment": metadata[idx]["내용"], "score": final_scores[idx]} for idx in sorted_indices]
    return results

# 데이터셋 로드
with open('national_heritage_qa_dataset_converted.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

retrieval_results = []
samples = []

for item in data:
    query = item["question"]
    ground_truth = item["ground_truth"]

    try:
        retrieved_documents = hybrid_search(query, top_k=5)
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
    except Exception as e:
        print(f"검색 실패: {e}")

# 검색 결과를 JSON 파일로 저장
with open("retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(retrieval_results, f, ensure_ascii=False, indent=4)

# 검색 결과를 CSV 파일로 저장
retrieval_results_df = pd.DataFrame(retrieval_results)
retrieval_results_df.to_csv("retrieval_results.csv", index=False, encoding="utf-8-sig")

# RAGAS 데이터셋 생성
dataset = EvaluationDataset(samples=samples)

metrics = [ContextPrecision()]
results = evaluate(dataset=dataset, metrics=metrics, show_progress=True)

# 평가 결과를 JSON 파일로 저장
results.to_json("evaluation_results.json")

# 평가 결과를 CSV 파일로 저장
results_df = results.to_pandas()
results_df.to_csv("evaluation_results.csv", index=False, encoding="utf-8-sig")

print("결과가 JSON 및 CSV 파일로 저장되었습니다.")
print("평가 결과:")
print(results)
