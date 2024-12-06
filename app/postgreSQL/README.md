# PostgreSQL

이 프로젝트는 PostgreSQL 데이터베이스를 사용하여 한국 국가유산 데이터를 처리하고 관리하는 다양한 스크립트를 포함하고 있습니다. 이 스크립트들은 데이터를 삭제, 삽입, 전처리 및 평가 데이터셋을 생성하는 과정을 포함하며, 주로 데이터베이스 관리와 데이터 전처리에 중점을 두고 있습니다.

## 📂 프로젝트 구조

```
📁 postgresql/
│
├── 📄 drop_national_heritage.py           # PostgreSQL 데이터베이스에서 기존 테이블 삭제 스크립트
├── 📄 fetch_and_preprocess.py             # 특정 데이터 조회 및 전처리 수행 스크립트
├── 📄 push_data_with_csv.py               # CSV 파일을 PostgreSQL 데이터베이스에 삽입하는 스크립트
├── 📄 rag_evaluation_dataset.py           # RAG 평가를 위한 데이터셋 생성 스크립트
├── 📄 national_heritage_sentences.csv     # 전처리된 국가유산 데이터 문장별 파일
├── 📄 updated01_national_heritage_full_data.csv # 원본 국가유산 데이터 파일
├── 📄 updated01_national_heritage_utf8.csv      # UTF-8로 인코딩된 국가유산 데이터 파일
└── 📄 README.md                          # 프로젝트 설명 파일
```

## 📝 주요 파일 설명

- **drop_national_heritage.py**: 
  - PostgreSQL 데이터베이스에서 기존의 'national_heritage' 테이블을 삭제합니다.

- **fetch_and_preprocess.py**: 
  - 특정 국가유산 데이터를 데이터베이스에서 가져와 전처리합니다. KSS 라이브러리를 사용하여 문장을 분리하고 구두점을 제거합니다.

- **push_data_with_csv.py**: 
  - CSV 파일로부터 데이터를 읽어와 PostgreSQL 데이터베이스에 삽입하는 스크립트입니다. 필요한 경우, 테이블을 생성하고 데이터베이스에 삽입하기 위해 각 데이터를 처리합니다.

- **rag_evaluation_dataset.py**: 
  - 데이터베이스에서 모든 데이터를 가져와 문장 단위로 분리하고, 평가를 위한 데이터셋을 CSV 파일로 저장합니다. 이 스크립트는 RAG 평가에서 사용할 수 있는 형태로 데이터를 준비하는 역할을 합니다.

- **national_heritage_sentences.csv**: 
  - `rag_evaluation_dataset.py`에서 생성된 파일로, 국가유산 데이터의 각 문장을 문장 ID와 함께 저장한 파일입니다.

- **updated01_national_heritage_full_data.csv**: 
  - 국가유산 데이터의 원본 CSV 파일입니다.

- **updated01_national_heritage_utf8.csv**: 
  - `push_data_with_csv.py` 스크립트에서 사용하기 위해 UTF-8로 인코딩된 CSV 파일입니다.

## ⚙️ 사용 방법

1. **데이터 삭제**: 
   - 데이터베이스의 기존 테이블을 삭제하려면 `drop_national_heritage.py` 스크립트를 실행하세요.

2. **데이터 삽입**: 
   - `push_data_with_csv.py`를 실행하여 국가유산 데이터를 PostgreSQL 데이터베이스에 삽입합니다. 삽입 전, 테이블이 존재하지 않을 경우 자동으로 생성됩니다.

3. **데이터 전처리**: 
   - 특정 데이터를 전처리하려면 `fetch_and_preprocess.py`를 실행하세요. 이 스크립트는 데이터를 문장 단위로 분리하고 전처리된 결과를 출력합니다.

4. **평가 데이터셋 생성**: 
   - `rag_evaluation_dataset.py`를 실행하여 데이터베이스에 있는 데이터를 문장 단위로 분리하고 CSV 파일로 저장하세요.

## 📌 주의 사항
- PostgreSQL 데이터베이스에 연결하기 위한 설정이 필요합니다. 각 스크립트에서 `host`, `database`, `user`, `password`를 알맞게 수정하세요.
- CSV 파일의 경로가 정확해야 합니다. 모든 CSV 파일은 `utf-8` 인코딩을 사용해야 합니다.
