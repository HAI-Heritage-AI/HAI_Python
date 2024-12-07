import os
import json
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.evaluation import evaluate
from ragas.metrics._context_entities_recall import ContextEntityRecall
import pandas as pd
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# JSON 파일 읽기
with open("retrieval_results.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 데이터 변환
samples = []
for item in raw_data:
    samples.append(
        SingleTurnSample(
            user_input=item["query"],
            response=None,  # 응답은 None으로 설정
            reference=item["ground_truth"],
            retrieved_contexts=[doc["text_segment"] for doc in item["retrieved_documents"]]
        )
    )

# 평가 데이터셋 생성
dataset = EvaluationDataset(samples=samples)

# Context Entities Recall 메트릭 정의
metrics = [ContextEntityRecall()]

# 평가 실행
results = evaluate(
    dataset=dataset,
    metrics=metrics,
    show_progress=True
)

# 결과를 pandas 데이터프레임으로 변환
final_results = results.to_pandas()

# JSON 저장
json_output_file = "context_entities_recall_evaluation_results.json"
final_results.to_json(json_output_file, orient="records", force_ascii=False, indent=4)
print(f"최종 평가 결과가 {json_output_file}에 저장되었습니다.")

# CSV 저장
csv_output_file = "context_entities_recall_evaluation_results.csv"
final_results.to_csv(csv_output_file, index=False, encoding="utf-8-sig")
print(f"최종 평가 결과가 {csv_output_file}에 저장되었습니다.")

# 결과 출력
print("Context Entities Recall 결과:")
print(results)
