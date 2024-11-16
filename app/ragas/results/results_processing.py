import json

# JSON 파일 읽기
with open("evaluation_results_ContextPrecision_pandas.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# context_precision 값 데이터 값 별로 필터링
# 정상적으로 처리가 되어 context_precision 값이 0이 아닌 것들
filtered_data = [item for item in data if item.get("context_precision") is not None and item["context_precision"] > 0]

# 비정상적으로 처리가 되어 context_precision 값이 0인 것들
zero_data = [item for item in data if item.get("context_precision") is None]

# 비정상적으로 처리가 되어 context_precision 값이 None인 것들
none_data = [item for item in data if item.get("context_precision") is item["context_precision"] == 0]

# 평균 계산
if filtered_data:
    average_precision1 = sum(item["context_precision"] for item in filtered_data) / len(filtered_data)
    average_precision2 = sum(item["context_precision"] for item in filtered_data) / (len(filtered_data) + len(zero_data))
    average_precision3 = sum(item["context_precision"] for item in filtered_data) / (len(filtered_data) + len(zero_data) + len(none_data))
else:
    average_precision = 0
    print(f"평균을 계산하는 도중 오류 발생: filtered_data가 없습니다. \nfiltered_data의 길이: {len(filtered_data)}")

# 결과 출력
print(f"0이상의 context_precision 평균: {average_precision1:.4f}")
print(f"0을 포함한 context_precision 평균: {average_precision2:.4f}")
print(f"모든 context_precision 평균: {average_precision3:.4f}")
print(f"전체 데이터 수: {len(filtered_data)+len(zero_data) + len(none_data)}")
print(f"제외된 데이터 수: {len(zero_data) + len(none_data)}")


excluded_data = zero_data + none_data

# 제외된 데이터 저장
excluded_file = "evaluation_results_ContextPrecision_pandas_excluded_data.json"
with open(excluded_file, "w", encoding="utf-8") as f:
    json.dump(excluded_data, f, ensure_ascii=False, indent=4)

print(f"제외된 데이터가 '{excluded_file}' 파일에 저장되었습니다.")
