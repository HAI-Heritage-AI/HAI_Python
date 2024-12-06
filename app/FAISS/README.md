# Chatbot Project

이 프로젝트는 faiss 경로에 있는 인덱스를 활용한 RAG(Retrieval-Augmented Generation) 기반 챗봇 관련 코드 모음입니다. 이 경로의 코드들은 주로 챗봇 시스템의 성능을 높이기 위한 인덱스 생성 및 메타데이터 처리에 중점을 두고 있습니다. 사용자가 입력한 질문을 임베딩하고, faiss 인덱스를 통해 유사한 정보를 검색하여 더 나은 답변 생성을 가능하게 합니다. 이 프로젝트는 한국 문화유산 관련 대화에 특화되어 있습니다.

## 📂 프로젝트 구조

```
📁 faiss/
│
├── 📂 archive/                # 📦 과거 코드와 자료들을 모아둔 폴더 (아직 유용할 수 있는 오래된 파일들 보관)
│
├── 📂 Index/                   # 📦 인덱스 파일 보관
│   ├── 📂 archive/             # 📦 이전에 생성된 인덱스 파일 보관
│   └── 📄 jhgan_cosine_index.bin  # 코사인 유사도 기반 인덱스 파일
│   └── 📄 jhgan_dotProduct_index.bin  # 내적 기반 인덱스 파일
│   └── 📄 jhgan_euclidean_index.bin  # 유클리드 거리 기반 인덱스 파일
│
├── 📂 Metadata/                # 📁 메타데이터 관련 폴더
│   ├── 📂 __pycache__/          # 🗂️ 파이썬 캐시 파일 (자동 생성된 임시 파일)
│   ├── 📂 archive/             # 📦 이전에 사용된 메타데이터 파일 보관
│   ├── 📄 jhgan_metadata.pkl   # 현재 메타데이터 파일
│   ├── 📄 read_metadata.py     # 🧾 메타데이터 피클 파일을 읽고 NaN 값을 식별하는 스크립트
│   └── 📄 temp_ner.py          # 🧠 한국어 NER(Named Entity Recognition)을 수행하는 스크립트
│
├── 📄 embedding.py             # 🧩 PostgreSQL에서 데이터를 가져와 임베딩하고 FAISS 인덱스를 생성하는 스크립트
│
└── 📄 README.md                # 📘 프로젝트 설명 파일
```

## 📝 주요 파일 설명

- **archive/**: 과거의 코드와 자료들을 보관하는 폴더로, 아직 유용할 수 있는 파일들을 모아둠.
- **Index/archive/**: 이전에 생성된 인덱스 파일들을 보관하는 폴더.
- **Metadata/archive/**: 이전에 사용된 메타데이터 파일들을 모아둔 폴더.
- **embedding.py**: PostgreSQL에서 데이터를 가져와 임베딩을 생성하고, FAISS 인덱스를 생성하는 역할을 하는 스크립트.
- **read_metadata.py**: 메타데이터 피클 파일을 읽어오고 NaN 값이 있는 행을 식별하는 스크립트.
- **temp_ner.py**: 메타데이터에서 첫 번째 텍스트에 대해 한국어 NER(Named Entity Recognition)을 수행하는 스크립트.
