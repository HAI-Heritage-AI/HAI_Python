import json

# JSON 파일 읽기
with open("context_recall_evaluation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# context_recall 값 데이터 값 별로 필터링
filtered_data = [item for item in data if item.get("context_recall", 0) > 0]
zero_data = [item for item in data if item.get("context_recall", 0) == 0]

# 평균 계산
if filtered_data:
    average_precision1 = sum(item["context_recall"] for item in filtered_data) / len(filtered_data)
    average_precision2 = sum(item["context_recall"] for item in filtered_data) / (len(filtered_data) + len(zero_data))
else:
    average_precision1 = average_precision2 = 0

# 결과 출력
print(f"0이상의 context_recall 평균: {average_precision1:.4f}")
print(f"0을 포함한 context_recall 평균: {average_precision2:.4f}")
print(f"전체 데이터 수: {len(filtered_data) + len(zero_data)}")
print(f"context_recall 값이 0인 데이터 수: {len(zero_data)}")
print(f"context_recall 값이 0이 아닌 데이터 수: {len(filtered_data)}")

# 제외된 데이터 저장 (0인 데이터)
excluded_file = "context_recall_evaluation_results_excluded_data.json"
with open(excluded_file, "w", encoding="utf-8") as f:
    json.dump(zero_data, f, ensure_ascii=False, indent=4)

print(f"제외된 데이터가 '{excluded_file}' 파일에 저장되었습니다.")
