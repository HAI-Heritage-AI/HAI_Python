# from rank_bm25 import BM25Okapi
# from konlpy.tag import Okt

# # Okt 토크나이저 초기화
# okt = Okt()

# # 문서 리스트 예시
# documents = ["한국에서 가장 오래된 목조 건축물은 무엇인가요?", 
#              "서울 숭례문은 1398년에 건립된 대표적인 문화재입니다."]

# # 문서들에 대해 Mecab으로 토큰화
# tokenized_documents = [okt.morphs(doc) for doc in documents]

# # BM25 인덱스 생성
# bm25 = BM25Okapi(tokenized_documents)

# # 예시 쿼리
# query = "한국에서 가장 오래된 건축물"

# # 쿼리를 Mecab으로 토큰화
# query_tokens = okt.morphs(query)

# # BM25 점수 계산
# bm25_scores = bm25.get_scores(query_tokens)

# # 결과 출력
# sorted_indices = np.argsort(-bm25_scores)[:3]
# for i in sorted_indices:
#     print(f"문서: {documents[i]} | 점수: {bm25_scores[i]:.4f}")

# from eunjeon import Mecab

# # Mecab 형태소 분석기 초기화
# mecab = Mecab()

# # 토큰화 함수 (허용할 품사만 남기고 나머지는 제외)
# def tokenize_with_mecab(text):
#     tokens = mecab.pos(text)  # 형태소 분석 후 품사 태깅
#     # 허용할 품사: NNP, NNG, NP, VV, VA, VCP, VCN, VSV, MAG, MAJ
#     allowed_pos_tags = ['NNP', 'NNG', 'NP', 'VV', 'VA', 'VCP', 'VCN', 'VSV', 'MAG', 'MAJ']
#     filtered_tokens = [word for word, pos in tokens if pos in allowed_pos_tags]
#     return filtered_tokens

# # 예제
# text = "한국에서 가장 오래된 목조 건축물은 무엇인가요?"
# tokens = tokenize_with_mecab(text)
# print(f"Mecab으로 품사 기반 토큰화 결과 : {tokens}")

# from gensim.models import KeyedVectors
# from huggingface_hub import hf_hub_download
# from eunjeon import Mecab

# # 1. Mecab 형태소 분석기 초기화
# mecab = Mecab()

# # 2. Hugging Face에서 nlpl_55 모델 다운로드 및 로드
# model_path = hf_hub_download(repo_id="Word2vec/nlpl_55", filename="model.bin")
# model = KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors="ignore")

# # 3. 동의어 추출 함수
# def get_synonyms(word, model, topn=5):
#     """
#     주어진 단어에 대해 Word2Vec 모델을 사용하여 가장 유사한 단어를 추출합니다.
#     """
#     try:
#         synonyms = model.most_similar(word, topn=topn)
#         return [synonym[0] for synonym in synonyms]
#     except KeyError:
#         return []  # 모델에 단어가 없을 경우 빈 리스트 반환

# # 4. 품사 기반으로 토큰화 및 동의어 확장
# def tokenize_with_mecab(text):
#     tokens = mecab.pos(text)  # 형태소 분석 후 품사 태깅
#     # 허용할 품사: NNP, NNG, NP, VV, VA, VCP, VCN, VSV, MAG, MAJ
#     allowed_pos_tags = ['NNP', 'NNG', 'NP', 'VV', 'VA', 'VCP', 'VCN', 'VSV', 'MAG', 'MAJ']
#     filtered_tokens = [word for word, pos in tokens if pos in allowed_pos_tags]
#     return filtered_tokens

# def expand_synonyms(text, model):
#     """
#     문장을 Mecab으로 토큰화하고, 각 토큰에 대해 동의어를 확장합니다.
#     """
#     # Mecab으로 토큰화
#     tokens = tokenize_with_mecab(text)

#     expanded_tokens = []
#     original_tokens = []

#     # 각 토큰에 대해 동의어를 확장
#     for token in tokens:
#         original_tokens.append(token)
#         synonyms = get_synonyms(token, model)
#         if synonyms:
#             expanded_tokens.append(synonyms)
#         else:
#             expanded_tokens.append([token])  # 동의어가 없으면 원본 단어 그대로

#     return original_tokens, expanded_tokens

# # 5. 테스트
# text = "한국에서 가장 오래된 목조 건축물은 무엇인가요?"
# original_tokens, expanded_tokens = expand_synonyms(text, model)

# print("원본 문장:", text)
# print("\n원래의 토큰과 동의어 확장 결과:")
# for original, expanded in zip(original_tokens, expanded_tokens):
#     print(f"원본: {original} -> 동의어: {expanded}")

# from konlpy.tag import Okt

# # Okt 형태소 분석기 초기화
# okt = Okt()

# # 토큰화 및 품사 태깅
# def tokenize_with_okt(text):
#     # 품사 태깅
#     tokens = okt.pos(text)
#     # 허용할 품사: 명사, 동사, 형용사 등
#     allowed_pos_tags = ['Noun', 'Verb', 'Adjective']
#     filtered_tokens = [word for word, pos in tokens if pos in allowed_pos_tags]
#     return filtered_tokens

# # 예제
# text = "한국에서 가장 오래된 목조 건축물은 무엇인가요?"
# tokens = tokenize_with_okt(text)
# print(f"OkT으로 품사 기반 토큰화 결과 : {tokens}")


from PyDictionary import PyDictionary

# PyDictionary 초기화
dictionary = PyDictionary()

# 예시 단어
word = "가장"

# 동의어 찾기
synonyms = dictionary.synonym(word)

# 결과 출력
print(f"{word}의 동의어는: {synonyms}")
