# Chatbot Project

이 프로젝트는 FAISS 인덱스를 활용한 RAG(Retrieval-Augmented Generation) 기반 챗봇입니다. 사용자가 입력한 질문을 임베딩하고, FAISS 인덱스를 통해 유사한 정보를 검색한 후, GPT-3.5를 통해 자연스럽고 풍부한 답변을 생성합니다. 이 프로젝트는 한국 문화유산 관련 대화에 특화되어 있습니다.

## 📂 프로젝트 구조

```
📁 chatbot/
│
├── 📂 archive/                # 📦 과거 코드와 자료들을 모아둔 폴더 (아직 유용할 수 있는 오래된 파일들 보관)
│
├── 📂 __pycache__/             # 🗂️ 파이썬 캐시 파일 (자동 생성된 임시 파일)
│
├── 📄 .env                     # 🔑 환경 변수 파일 (API 키 등 비밀 정보 관리)
│
├── 📄 chat_with_faiss.py       # 🤖 FAISS 인덱스를 활용하여 RAG 기반으로 답변을 생성하는 챗봇 스크립트
│
├── 📄 chat_with_hybrid.py      # 🔍 BM25 및 FAISS를 결합한 하이브리드 검색을 통해 답변을 생성하는 챗봇
│
├── 📄 hybrid_search.py         # 🧩 BM25와 FAISS를 활용해 다양한 방법으로 검색하는 하이브리드 검색 기능 구현 스크립트
│
└── 📄 README.md                # 📘 프로젝트 설명 파일
```

## 📝 주요 파일 설명

- **archive/**: 과거의 코드와 자료들을 보관하는 폴더로, 아직 유용할 수 있는 파일들을 모아둠.
- **chat_with_faiss.py**: FAISS로 질문을 검색하고 GPT-3.5로 답변 생성.
- **chat_with_hybrid.py**: BM25와 FAISS를 결합하여 질문에 대한 답변 생성.
- **hybrid_search.py**: BM25와 FAISS를 사용해 하이브리드 검색 기능 구현.
- **.env**: 환경 변수 파일 (API 키 등 비밀 정보 관리).
