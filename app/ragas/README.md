# RAGAS Evaluation Project

이 프로젝트는 RAG 기반의 검색 및 QA 시스템 성능 평가를 위한 다양한 스크립트와 데이터 파일들을 포함하고 있습니다. 이 경로의 코드들은 주로 검색 시스템에서의 precision, recall 등 평가 지표를 측정하고 분석하는 데 중점을 두고 있습니다.

## 📂 프로젝트 구조

```
📁 ragas/
│
├── 📂 results/                # 평가 결과와 관련된 파일들이 저장된 폴더
│   ├── 📄 context_entities_recall_evaluation.py      # 컨텍스트 내 엔티티 recall 평가 스크립트
│   ├── 📄 context_precision_evaluation.py            # 컨텍스트 precision 평가 스크립트
│   ├── 📄 context_recall_evaluation.py               # 컨텍스트 recall 평가 스크립트
│   ├── 📄 national_heritage_qa_dataset_converted.json # 국가유산 QA 데이터셋 (변환된 형식)
│   ├── 📄 NonLLMContextPrecisionWithReferenc.py      # LLM을 사용하지 않는 컨텍스트 precision 평가 스크립트
│   └── 📄 retrieval_results.json                     # 검색 결과가 저장된 JSON 파일
```

## 📝 주요 파일 설명

- **context_entities_recall_evaluation.py**:
  - 검색된 결과에서 컨텍스트 내 엔티티의 recall을 평가하는 스크립트입니다. QA 시스템이 얼마나 잘 엔티티를 회수했는지 측정합니다.

- **context_precision_evaluation.py**:
  - 검색된 결과에서 컨텍스트의 precision을 평가하는 스크립트입니다. QA 시스템이 반환한 컨텍스트가 얼마나 정확한지 평가합니다.

- **context_recall_evaluation.py**:
  - 검색된 결과에서 컨텍스트의 recall을 평가하는 스크립트입니다. 주어진 질문에 대해 시스템이 얼마나 관련된 정보를 잘 회수했는지 확인합니다.

- **national_heritage_qa_dataset_converted.json**:
  - 국가유산 QA 데이터셋 파일로, 변환된 형식을 사용하여 RAG 기반 검색을 테스트하고 평가하는 데 사용됩니다.

- **NonLLMContextPrecisionWithReferenc.py**:
  - LLM 모델을 사용하지 않고 검색된 컨텍스트의 precision을 평가하는 스크립트입니다. 비 LLM 기반 접근의 평가를 수행합니다.

- **retrieval_results.json**:
  - 검색된 문서들과 질문, 참조 문서들에 대한 정보를 저장한 결과 파일입니다. 검색 결과를 후속 평가에 사용할 수 있도록 저장합니다.

## 📌 사용 방법

1. **평가 수행**:
   - 평가하고자 하는 지표에 따라 적절한 스크립트를 실행합니다. 예를 들어, 컨텍스트 recall 평가를 원할 경우 `context_recall_evaluation.py`를 사용하세요.

2. **데이터 파일 준비**:
   - 평가를 수행하기 위해 `national_heritage_qa_dataset_converted.json`과 같은 데이터 파일이 필요합니다. 이 파일은 국가유산 관련 QA 데이터셋으로 변환된 형식을 사용합니다.

3. **검색 결과 저장**:
   - 검색 결과는 `retrieval_results.json`에 저장되며, 평가 스크립트들이 이 파일을 활용해 성능을 측정합니다.

## ⚠️ 주의 사항
- 모든 스크립트를 실행하기 전에 Python 환경이 적절히 설정되어 있는지 확인하세요.
- 평가를 위해서는 적절한 형식의 데이터셋이 필요합니다. 데이터 파일 경로가 올바른지 확인해 주세요.