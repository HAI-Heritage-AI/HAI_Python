### embedding_1000.py
>  PostgreSQL 데이터베이스에서 텍스트 데이터를 1000개 가져와 임베딩을 생성하고, 이를 FAISS 인덱스와 메타데이터로 저장

### embedding_1000_processing.py
> PostgreSQL 데이터베이스에서 텍스트 데이터를 1000개 가져와서 문장별로 분리하여 임베딩을 생성하고, 이를 FAISS 인덱스와 메타데이터로 저장

### embedding_cosine.py
> 전처리 한 문장들을 코사인 유사도로 검색 가능하게 임베딩 생성

### embedding_uclid.py
> 전처리 한 문장들을 유클리드 거리로 검색 가능하게 임베딩 생성